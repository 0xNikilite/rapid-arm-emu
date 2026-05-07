//! ## Problem brief
//!
//! We need to insert halt checks often enough that execution cannot run
//! forever without giving the runtime a chance to stop it.
//!
//! The basic invariant is counted in safepoints, not in blocks or
//! instructions. If `halt_check_every = N`, then after every `N` safepoints
//! on any executable path, the next safepoint must be followed by a halt
//! check.
//!
//! There are two separate problems:
//!
//! 1. **Acyclic path problem**: on a DAG, make sure long paths with many
//!    safepoints get periodic halt checks.
//! 2. **Cycle problem**: on a general directed CFG, make sure no directed
//!    cycle can run forever without executing a halt check.
//!
//! The DAG problem can be solved with a forward countdown state.
//!
//! The cycle problem needs an additional structural invariant: every directed
//! cycle must contain at least one safepoint. Once that is true, we can force
//! halt checks inside cyclic SCCs.
//!
//! ---
//!
//! ## Invariants
//!
//! `halt_check_in = r` means:
//!
//! - after seeing `r` more safepoints, the next safepoint must be followed by
//!   a halt check.
//!
//! Equivalently, if `N = halt_check_every`, then this path has seen
//! `N - r` safepoints since the previous halt check.
//!
//! The state carried across edges is:
//!
//! ```nocompile_test
//! struct HaltState {
//!     remaining: NonZero<usize>,
//!
//!     // Safepoints since the previous halt check on this edge/path.
//!     //
//!     // This must be edge/path-local. It may contain safepoints from multiple
//!     // blocks, not only the immediate predecessor block.
//!     suffix_safepoints: Queue<SafepointLoc>,
//! }
//! ````
//!
//! When processing a safepoint:
//!
//! ```nocompile_test
//! suffix_safepoints.push(loc);
//! remaining -= 1;
//!
//! if remaining == 0 {
//!     insert_halt_check_after(loc);
//!     suffix_safepoints.clear();
//!     remaining = N;
//! }
//! ```
//!
//! At a branch, clone the state to each successor.
//!
//! ---
//!
//! ## DAG solution
//!
//! For a DAG, process blocks in topological order.
//!
//! For merges, the simplest correct rule is `min`.
//!
//! ```nocompile_test
//! halt_check_in_at_merge = predecessors.iter().map(|predecessor| {
//!     predecessor.halt_check_out
//! }).min().unwrap();
//! ```
//!
//! Why?
//!
//! If one predecessor arrives with:
//!
//! ```text
//! block1 out = 8
//! block2 out = 3
//! ```
//!
//! then choosing `3` means the continuation may insert a halt check earlier
//! on the `block1` path than strictly necessary, but it will never insert one
//! too late.
//!
//! That gives a clean first implementation:
//!
//! ```nocompile_test
//! entry halt_check_in = N
//!
//! for block in topo_order {
//!     let halt_check_in = if block == ENTRYPOINT {
//!         N
//!     } else {
//!         min(pred.halt_check_out for pred in preds(block))
//!     };
//!
//!     let halt_check_out = process_block(block, halt_check_in);
//!
//!     for succ in succs(block) {
//!         edge_state[block -> succ] = halt_check_out;
//!     }
//! }
//! ```
//!
//! This is conservative, deterministic, and easy to prove correct.
//!
//! ---
//!
//! ## Merge optimization for large countdown differences
//!
//! The `min` rule is always safe, but can be conservative.
//!
//! Suppose a merge has two incoming countdowns:
//!
//! ```text
//! r_hi = max incoming remaining
//! r_lo = min incoming remaining
//! ```
//!
//! where:
//!
//! ```text
//! r_hi > r_lo
//! ```
//!
//! If we want to normalize the `r_lo` edge up to `r_hi`, then we insert a
//! halt check on the `r_lo` path before the merge.
//!
//! The correct location is:
//!
//! ```text
//! after the safepoint that has exactly N - r_hi safepoints after it before the merge
//! ```
//!
//! Equivalently:
//!
//! ```text
//! after the (r_hi - r_lo)-th safepoint since the previous halt check
//! ```
//!
//! So if we keep a suffix list of safepoints since the last halt check, the
//! insertion point is:
//!
//! ```nocompile_test
//! let target = r_hi;
//!
//! let safepoints_after_insert = N - target;
//!
//! let insertion_safepoint = suffix_safepoints
//!     .iter()
//!     .nth_back(safepoints_after_insert);
//! ```
//!
//! This optimization is only valid when the edge has a complete path-local
//! unchecked suffix. If suffix precision was lost at a previous merge, or if
//! the needed safepoint is not present in the suffix, fall back to `min`.
//!
//! ---
//!
//! ## Why the insertion point may not be in the immediate predecessor
//!
//! Let:
//!
//! ```text
//! N = 10
//! ```
//!
//! CFG:
//!
//! ```text
//! entry:
//!     br cond A B1
//!
//! A:
//!     safepoint
//!     safepoint
//!     safepoint
//!     br M
//!
//! B1:
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     br B2
//!
//! B2:
//!     safepoint
//!     safepoint
//!     br M
//!
//! M:
//!     ...
//! ```
//!
//! Assume both branches start with:
//!
//! ```text
//! halt_check_in = 10
//! ```
//!
//! Then:
//!
//! ```text
//! A has 3 safepoints => remaining = 7
//! B has 9 safepoints => remaining = 1
//! ```
//!
//! At merge `M`:
//!
//! ```text
//! r_hi = 7
//! r_lo = 1
//! diff = 6
//! ```
//!
//! If we normalize the `B` path to `7`, we need to insert a halt check so
//! that there are:
//!
//! ```text
//! N - r_hi = 10 - 7 = 3
//! ```
//!
//! safepoints after the inserted check before the merge.
//!
//! But `B2` contains only 2 safepoints.
//!
//! So the correct insertion point is in `B1`, not in the immediate
//! predecessor `B2`.
//!
//! Specifically, the `B` suffix has 9 safepoints total:
//!
//! ```text
//! B1: s1 s2 s3 s4 s5 s6 s7
//! B2: s8 s9
//! ```
//!
//! We need 3 safepoints after the inserted check, so we insert after `s6`:
//!
//! ```text
//! B1: s1 s2 s3 s4 s5 s6 HALT_CHECK s7
//! B2: s8 s9
//! ```
//!
//! Then after the inserted halt check, the path sees 3 safepoints before the
//! merge, so it arrives with:
//!
//! ```text
//! 10 - 3 = 7
//! ```
//!
//! matching the `A` path.
//!
//! Therefore the immediate predecessor block is not enough. The optimization
//! needs path-local suffix history.
//!
//! ---
//!
//! ## General directed graph solution
//!
//! A general CFG may contain cycles, so ordinary block topological order is
//! not enough.
//!
//! The solution is to work over strongly connected components.
//!
//! ### Step 1: Validate safepoint coverage of cycles
//!
//! Every directed cycle must contain at least one safepoint.
//!
//! Equivalently:
//!
//! ```text
//! the subgraph induced by blocks with no safepoints must be acyclic
//! ```
//!
//! Why this equivalence holds:
//!
//! * If there is a directed cycle with no safepoint, every block on that cycle
//!   is safepoint-free, so the safepoint-free subgraph contains a cycle.
//! * If the safepoint-free subgraph contains a cycle, then the original graph
//!   has a directed cycle with no safepoint.
//!
//! So the pass should compute SCCs of the safepoint-free subgraph. If any SCC
//! is cyclic, the IR invariant is violated. Either an earlier pass must insert
//! a safepoint into that cycle, or this pass must fail loudly.
//!
//! ### Step 2: Condense SCCs into a DAG
//!
//! Compute SCCs of the full CFG.
//!
//! The SCC condensation graph is always a DAG, even if the original CFG has
//! cycles.
//!
//! Process SCCs in topological order.
//!
//! ### Step 3: Process acyclic SCCs with the DAG countdown logic
//!
//! An acyclic SCC is a single block with no self-loop.
//!
//! For these components, use the ordinary `HaltState` transfer:
//!
//! ```nocompile_test
//! halt_state_in = merge predecessor states
//! halt_state_out = process_block(block, halt_state_in)
//! ```
//!
//! This preserves the existing DAG behavior.
//!
//! ### Step 4: Process cyclic SCCs by forcing checks after safepoints
//!
//! For a cyclic SCC, we already know every cycle contains at least one
//! safepoint.
//!
//! Therefore a simple safe rule is:
//!
//! ```text
//! insert a halt check after every safepoint in the cyclic SCC
//! ```
//!
//! This guarantees every directed cycle in the SCC contains at least one halt
//! check.
//!
//! This may insert more halt checks than the theoretical minimum, but it is
//! simple, local, and sound. Computing a smaller set of safepoints that hits
//! every cycle is a feedback-vertex-style optimization and should not be the
//! first implementation.
//!
//! ### Step 5: Conservatively summarize cyclic SCC exits
//!
//! After forcing halt checks inside a cyclic SCC:
//!
//! * paths that hit a safepoint reset their countdown to `N`;
//! * paths that do not hit a safepoint preserve the incoming countdown.
//!
//! Since every `remaining` value is `<= N`, the safe summary for every exit is
//! the incoming merged countdown with suffix history discarded:
//!
//! ```nocompile_test
//! halt_state_out = HaltState {
//!     remaining: halt_state_in.remaining,
//!     suffix_safepoints: Queue::new(),
//! };
//! ```
//!
//! Discarding suffix precision is important because inside a cyclic SCC there
//! may be many possible paths. Keeping one concrete suffix would be unsound.
//!
//! ---
//!
//! ## Final strategy
//!
//! For arbitrary directed graphs:
//!
//! ```nocompile_test
//! assert every cycle has at least one safepoint
//!
//! compute SCCs
//! compute SCC condensation DAG
//!
//! for component in scc_topological_order {
//!     let halt_state_in = merge external predecessor states
//!
//!     if component is acyclic {
//!         process the single block with HaltState
//!         propagate halt_state_out to successors
//!     } else {
//!         insert a halt check after every safepoint in the component
//!
//!         // Conservative SCC summary.
//!         halt_state_out = HaltState::from_remaining(halt_state_in.remaining)
//!
//!         propagate halt_state_out to successors outside the component
//!     }
//! }
//! ```
//!
//! This gives:
//!
//! 1. the precise DAG behavior for acyclic code;
//! 2. safe periodic halt checks along long acyclic paths;
//! 3. guaranteed halt checks inside every cycle;
//! 4. no fixed-point countdown analysis inside loops;
//! 5. no unsound use of suffix history across cyclic control flow.

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZero;
use smallvec::SmallVec;
use crate::ir::{Block, ExecIrBuilder, Stmt, StmtKind};
use crate::ir::arena::{handle_impl_helper, make_handle, Arena, ArenaMap, ArenaSet, Storable};

#[derive(Copy, Clone)]
struct ShouldHalt(bool);


/// represents the state at the end of one path into the merge.
/// After seeing `r` more safepoints, the next safepoint forces a halt check.
/// Equivalently, if:
///
/// N = halt_check_every
///
/// then this path has already seen:
///
/// N - r
///
/// safepoints since the previous halt check.
#[derive(Clone)]
struct HaltState {
    remaining: NonZero<usize>,
    suffix_safepoints: rpds::Queue<Stmt>
}

struct HaltStateMap {
    outgoing_edges: ArenaMap<Block, HashMap<Block, HaltState>>,
    incoming_edges: ArenaMap<Block, HashSet<Block>>
}

impl HaltStateMap {
    pub fn new(ir: &ExecIrBuilder) -> Self {
        let edge_capacity = ir.blocks.len().div_ceil(2).saturating_mul(3);
        let forward_edges = ArenaMap::with_capacity(edge_capacity);
        let backward_edges = ArenaMap::with_capacity(edge_capacity);

        Self {
            outgoing_edges: forward_edges,
            incoming_edges: backward_edges
        }
    }

    pub fn add_edge(&mut self, from: Block, to: Block, state: HaltState) -> Option<HaltState> {
        let edges = {
            self.outgoing_edges.get_or_insert_with(from, || HashMap::with_capacity(1))
        };

        let backward_edges = {
            self.incoming_edges.get_or_insert_with(to, || HashSet::with_capacity(1))
        };

        backward_edges.insert(from);
        edges.insert(to, state)
    }

    pub fn drain_incoming(
        &mut self,
        towards: Block
    ) -> impl Iterator<Item=(Block, HaltState)> + use<'_> {
        struct Drain<'a> {
            towards: Block,
            outgoing_edges: &'a mut ArenaMap<Block, HashMap<Block, HaltState>>,
            drain: std::collections::hash_set::Drain<'a, Block>,
        }

        impl Iterator for Drain<'_> {
            type Item = (Block, HaltState);

            fn next(&mut self) -> Option<Self::Item> {
                self.drain.next().map(|from| {
                    let to = self.towards;
                    let edge_must_exist = "edge must exist in forward map if it exists in backward map";
                    let state = self.outgoing_edges
                        .get_mut(from)
                        .and_then(|map| map.remove(&to))
                        .expect(edge_must_exist);

                    (to, state)
                })
            }
        }

        impl Drop for Drain<'_> {
            fn drop(&mut self) {
                for _ in self {
                    // run next to completion
                }
            }
        }


        self
            .incoming_edges
            .get_mut(towards)
            .map(|incoming_edges| {
                Drain {
                    towards,
                    outgoing_edges: &mut self.outgoing_edges,
                    drain: incoming_edges.drain(),
                }
            })
            .into_iter()
            .flatten()
    }
}


struct HaltCheckInserter<'a> {
    ir: &'a mut ExecIrBuilder,
    // note: this is deliberately a hashmap
    //       because there aren't that many
    //       safepoints compared to other types of stmts
    safepoint_stmt_to_block_and_index: HashMap<Stmt, (Block, usize)>,
    map: HaltStateMap,
}

impl<'a> HaltCheckInserter<'a> {
    pub fn new(ir: &'a mut ExecIrBuilder) -> Self {
        let mut safepoints =
            HashMap::with_capacity(ir.stmts.len().div_ceil(128));

        for (block, data) in ir.blocks.iter() {
            for (i, stmt) in data.stmts.iter().copied().enumerate() {
                if let StmtKind::Safepoint = ir.stmts[stmt].rvalue {
                    let old_pos = safepoints.insert(stmt, (block, i));
                    assert!(old_pos.is_none());
                }
            }
        }

        let map = HaltStateMap::new(ir);

        Self {
            ir,
            safepoint_stmt_to_block_and_index: safepoints,
            map
        }
    }

    pub fn ir(&mut self) -> &mut ExecIrBuilder {
        self.ir
    }

    pub fn insert_halt_check_after_safepoint_indexed(
        &mut self,
        block: Block,
        stmt_index: usize,
    ) -> Block {
        let safepoints = &mut self.safepoint_stmt_to_block_and_index;

        let continuation = self.ir.insert_halt_check_at(block, stmt_index.strict_add(1));
        for (i, &stmt) in self.ir.blocks[continuation].stmts.iter().enumerate() {
            if let StmtKind::Safepoint = self.ir.stmts[stmt].rvalue {
                let old = safepoints.insert(stmt, (continuation, i));
                assert!(old.is_some_and(|(old_block, old_idx)| {
                    old_block == block && old_idx > stmt_index
                }))
            }
        }

        // since the edges are (from -> to) where HaltState is the state of things after
        // from runs to completion; when splitting a node, to is unaffected.
        // since it will always exist, and the edge from -> to always means
        // the end of `from` jumps towards `to`
        // and so we need to remap anything `from` maps to and place it in `continuation`
        if let Some(edges) = self.map.outgoing_edges.remove(block) {
            for &to in edges.keys() {
                let removed = self.map.incoming_edges.get_mut(to).unwrap().remove(&block);
                assert!(removed)
            }

            let old_edges = self.map.outgoing_edges.insert(continuation, edges);
            assert!(
                old_edges.is_none(),
                "continuation is a fresh block, and can't have any existing edges"
            );
        }


        continuation
    }

    // TODO better merger
    //
    // pub fn insert_halt_check_after_safepoint(&mut self, at: Stmt) -> (Block, Block) {
    //     let safepoints = &mut self.safepoint_stmt_to_block_and_index;
    //     let (block, stmt_index) = safepoints[&at];
    //     let continuation = self.insert_halt_check_after_safepoint_indexed(block, stmt_index);
    //     (block, continuation)
    // }
}


impl HaltState {
    fn from_remaining(remaining: NonZero<usize>) -> Self {
        Self {
            remaining,
            suffix_safepoints: rpds::Queue::new(),
        }
    }

    pub fn push_safepoint(
        &mut self,
        halt_check_every: NonZero<usize>,
        safepoint: Stmt,
    ) -> ShouldHalt {
        let new_remaining = NonZero::new(self.remaining.get().strict_sub(1));
        self.remaining = new_remaining.unwrap_or(halt_check_every);

        let halt = ShouldHalt(new_remaining.is_none());

        match halt {
            // There is never a need to keep suffix history before this point,
            // because after inserting a halt check the countdown resets to N.
            ShouldHalt(true) => self.suffix_safepoints = rpds::Queue::new(),
            ShouldHalt(false) => {
                self.suffix_safepoints.enqueue_mut(safepoint);
                if self.suffix_safepoints.len() > halt_check_every.get() {
                    self.suffix_safepoints.dequeue_mut();
                }
            }
        }

        halt
    }

    #[inline]
    fn break_down_block_inner(
        mut this: Option<&mut Self>,
        inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        mut block: Block
    ) -> Block {
        'break_down_loop: loop {
            let ir = inserter.ir();

            let mut split = None;
            for (i, &stmt) in ir.blocks[block].stmts.iter().enumerate() {
                if let StmtKind::Safepoint = ir.stmts[stmt].rvalue {
                    let should_halt = this.as_mut().map_or(
                        ShouldHalt(true),
                        |this| this.push_safepoint(
                            halt_check_every,
                            stmt
                        )
                    );

                    if should_halt.0 {
                        split = Some(i);
                        break
                    }
                }
            }

            let Some(pos) = split else {
                break 'break_down_loop
            };

            block = inserter.insert_halt_check_after_safepoint_indexed(block, pos);
        }

        block
    }

    pub fn break_down_block(
        mut self,
        ir: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        block: Block
    ) -> (Block, Self) {
        let last_block = Self::break_down_block_inner(
            Some(&mut self),
            ir,
            halt_check_every,
            block
        );

        (last_block, self)
    }

    /// For cyclic SCCs, countdown analysis inside the SCC is not necessary for
    /// termination safety. Once we have proven every cycle contains at least one
    /// safepoint, inserting a halt check after every safepoint in the cyclic SCC
    /// guarantees every cycle contains at least one halt check.
    pub fn force_halt_checks_after_safepoints(
        ir: &mut HaltCheckInserter,
        block: Block,
    ) -> Block {
        Self::break_down_block_inner(
            None,
            ir,
            const { NonZero::new(1).unwrap() },
            block
        )
    }

    fn merge_halt_states(
        _inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        incoming: &mut [HaltState],
    ) -> HaltState {
        match incoming {
            // No incoming edges; start fresh
            [] => HaltState::from_remaining(halt_check_every),

            // Single predecessor: preserve precise path-local suffix history.
            [state] => state.clone(),

            _ => {
                let r_lo = incoming
                    .iter()
                    .map(|state| state.remaining)
                    .min()
                    .unwrap();

                // TODO better merger
                // let r_hi = incoming
                //     .iter()
                //     .map(|state| state.remaining)
                //     .max()
                //     .unwrap();
                //
                // let diff = r_hi.get().strict_sub(r_lo.get());
                // let threshold = halt_check_every
                //     .div_ceil(const { NonZero::new(2).unwrap() });
                //
                // if diff > threshold.get() {
                //     let can_normalize = ...;
                //     if can_normalize {
                //         /*normalize*/
                //         return HaltState::from_remaining(r_hi);
                //     }
                // }

                HaltState::from_remaining(r_lo)
            }
        }
    }
}


make_handle!(SccComponent);

handle_impl_helper! {
    impl usize like for SccComponent;
}

struct OwnedComponent(Vec<Block>);

impl Storable for OwnedComponent {
    type Handle = SccComponent;
}

struct Tarjan<'a> {
    ir: &'a ExecIrBuilder,
    allowed: Option<&'a ArenaSet<Block>>,

    next_index: usize,
    index: ArenaMap<Block, usize>,
    lowlink: ArenaMap<Block, usize>,

    stack: Vec<Block>,
    on_stack: ArenaSet<Block>,

    components: Arena<OwnedComponent>,
}

impl<'a> Tarjan<'a> {
    pub fn strong_connect(&mut self, block: Block) {
        stacker::maybe_grow(
            4 * 1024,
            2 * 1024 * 1024,
            move || self.strong_connect_inner(block)
        )
    }

    #[inline(always)]
    fn block_is_allowed(&self, block: Block) -> bool {
        self.allowed.is_none_or(|allowed| allowed.contains(block))
    }


    fn strong_connect_inner(&mut self, block: Block) {
        self.index.insert(block, self.next_index);
        self.lowlink.insert(block, self.next_index);
        self.next_index = self.next_index.strict_add(1);

        self.stack.push(block);
        self.on_stack.insert(block);

        for succ in self.ir.successors(block) {
            if !self.block_is_allowed(succ) {
                continue
            }

            if self.index.get(succ).is_none() {
                self.strong_connect(succ);

                let block_lowlink = self.lowlink[block];
                let succ_lowlink = self.lowlink[succ];

                self.lowlink.insert(block, block_lowlink.min(succ_lowlink));
            } else if self.on_stack.contains(succ) {
                let block_lowlink = self.lowlink[block];
                let succ_index = self.index[succ];

                self.lowlink.insert(block, block_lowlink.min(succ_index));
            }
        }

        if self.lowlink[block] == self.index[block] {
            let mut component = Vec::new();

            loop {
                let member = self.stack.pop().unwrap();
                self.on_stack.remove(member);
                component.push(member);

                if member == block {
                    break
                }
            }

            self.components.store(OwnedComponent(component));
        }
    }

    fn run(mut self) -> Arena<OwnedComponent> {
        for block in self.ir.blocks.keys() {
            if !self.block_is_allowed(block) {
                continue
            }

            if self.index.get(block).is_none() {
                self.strong_connect(block);
            }
        }

        self.components
    }
}

fn strongly_connected_components(
    ir: &ExecIrBuilder,
    allowed: Option<ArenaSet<Block>>
) -> Arena<OwnedComponent> {
    let tarjan = Tarjan {
        ir,
        allowed: allowed.as_ref(),

        next_index: 0,
        index: ArenaMap::new(),
        lowlink: ArenaMap::new(),

        stack: Vec::new(),
        on_stack: ArenaSet::new(),

        components: Arena::new(),
    };

    tarjan.run()
}


fn block_has_safepoint(ir: &ExecIrBuilder, block: Block) -> bool {
    ir.blocks[block]
        .stmts
        .iter()
        .any(|&stmt| matches!(&ir.stmts[stmt].rvalue, StmtKind::Safepoint))
}

fn component_is_cyclic(ir: &ExecIrBuilder, component: &[Block]) -> bool {
    match *component {
        // No blocks: never cyclic.
        [] => false,

        // One block: cyclic only if it has a self-loop.
        [one_block] => ir.successors(one_block).any(|succ| succ == one_block),

        // A multi-block SCC is always cyclic.
        //
        // Pick any two distinct blocks A and B. Since this is an SCC,
        // A reaches B and B reaches A; concatenating those paths gives
        // a directed cycle.
        [_, _, ..] => true
    }
}


struct SccGraph {
    components: Arena<OwnedComponent>,
    component_of: ArenaMap<Block, SccComponent>,
    topo_order: Vec<SccComponent>,
    is_cyclic: ArenaSet<SccComponent>,
}

fn assert_cycles_have_safepoints(ir: &ExecIrBuilder) {
    let allowed = ir
        .blocks
        .keys()
        .filter(|&block| !block_has_safepoint(ir, block))
        .collect::<ArenaSet<_>>();

    let components = strongly_connected_components(ir, Some(allowed));

    for (_, component) in components.iter() {
        if component_is_cyclic(ir, &component.0) {
            panic!(
                "IR invariant violated: found a directed cycle with no safepoint"
            );
        }
    }
}

impl SccGraph {
    pub fn new(ir: &ExecIrBuilder) -> Self {
        assert_cycles_have_safepoints(ir);

        let components = strongly_connected_components(ir, None);
        let mut component_of = ArenaMap::<Block, SccComponent>::with_capacity(
            ir.blocks.len()
        );

        for (component_id, component) in components.iter() {
            for &block in &component.0 {
                component_of.insert(block, component_id);
            }
        }

        let component_of = component_of;

        let mut edges = vec![ArenaSet::<SccComponent>::new(); components.len()];
        let mut indegree = vec![0_usize; components.len()];

        for (block, &from_component) in component_of.iter() {
            for succ in ir.successors(block) {
                let to_component = component_of[succ];

                if from_component == to_component {
                    continue;
                }

                if edges[from_component.get()].insert(to_component) {
                    indegree[to_component.get()] = indegree[to_component.get()].strict_add(1);
                }
            }
        }

        let mut ready = indegree
            .iter()
            .enumerate()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(component_id, _)| SccComponent::new(component_id))
            .collect::<VecDeque<SccComponent>>();

        let mut topo_order = Vec::with_capacity(components.len());

        while let Some(component_id) = ready.pop_front() {
            topo_order.push(component_id);

            for succ_component in edges[component_id.get()].iter() {
                indegree[succ_component.get()] = indegree[succ_component.get()].strict_sub(1);

                if indegree[succ_component.get()] == 0 {
                    ready.push_back(succ_component);
                }
            }
        }

        assert_eq!(topo_order.len(), components.len());

        let is_cyclic = components
            .iter()
            .filter(|(_, component)| component_is_cyclic(ir, &component.0))
            .map(|(component, _)| component)
            .collect::<ArenaSet<_>>();

        SccGraph {
            components,
            component_of,
            topo_order,
            is_cyclic,
        }
    }

    pub fn component_is_cyclic(&self, component: SccComponent) -> bool {
        self.is_cyclic.contains(component)
    }
}


#[allow(dead_code)]
pub fn insert_halt_checks(ir: &mut ExecIrBuilder) {
    let halt_check_every = NonZero::<usize>::try_from(ir.halt_check_every)
        .unwrap_or(NonZero::<usize>::MAX);

    let scc_graph = SccGraph::new(ir);

    let mut inserter = HaltCheckInserter::new(ir);

    for component_id in scc_graph.topo_order.iter().copied() {
        let component = scc_graph.components[component_id].0.as_slice();


        // TODO more complex analysis to be able to have fewer halt checks
        //      this currently works though, and this is low priority for now
        //      note if this changes, please update the docs for this module
        if scc_graph.component_is_cyclic(component_id) {
            // We intentionally do not use precise incoming countdown state inside
            // a cyclic SCC. Every safepoint in the SCC gets a halt check, which
            // makes every cycle terminating-safe because cycles without safepoints
            // were rejected by SccGraph::new.
            for &comp in component {
                inserter
                    .map
                    .drain_incoming(comp)
                    .fold((), |(), item| drop(item))
            }

            for &block in component {
                let last_block = HaltState::force_halt_checks_after_safepoints(
                    &mut inserter,
                    block,
                );


                for succ in inserter.ir().successors(last_block) {
                    let succ_component = scc_graph.component_of[succ];

                    if succ_component == component_id {
                        continue;
                    }

                    // Conservative SCC boundary:
                    //
                    // There may be an entry -> exit path through this cyclic SCC
                    // that sees no safepoint. Therefore we cannot safely reset to
                    // `halt_check_every` on outgoing edges.
                    //
                    // Using remaining = 1 means the next safepoint after the SCC
                    // will get a halt check. This may over-insert, but it is safe.
                    let old = inserter.map.add_edge(
                        last_block,
                        succ,
                        HaltState::from_remaining(const { NonZero::new(1).unwrap() })
                    );

                    debug_assert!(old.is_none());
                }
            }

            continue;
        }

        // Non-cyclic SCCs are exactly singleton blocks with no self-loop.
        let &[block] = component else {
            unreachable!("non-cyclic SCC should be a singleton block");
        };

        let mut incoming = inserter.map.drain_incoming(block)
            .map(|(_predecessor, state)| state)
            .collect::<SmallVec<[_; 32]>>();

        let state = HaltState::merge_halt_states(
            &mut inserter,
            halt_check_every,
            &mut incoming,
        );

        let (last_block, state) = state.break_down_block(
            &mut inserter,
            halt_check_every,
            block,
        );

        for succ in inserter.ir().successors(last_block) {
            let old = inserter.map.add_edge(
                last_block,
                succ,
                state.clone(),
            );

            debug_assert!(old.is_none());
        }
    }
}