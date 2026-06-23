#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(16))]
pub struct Vector(pub u128);

const _: () = assert!(align_of::<Vector>() == 16 && size_of::<Vector>() == 16);

pub const X_REGISTER_COUNT: u8 = 31;

#[derive(bytemuck::Zeroable, Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct PState(pub u32);

impl PState {
    pub const NEGATIVE: Self = Self(1 << 31);
    pub const ZERO: Self = Self(1 << 30);
    pub const CARRY: Self = Self(1 << 29);
    pub const OVERFLOW: Self = Self(1 << 28);

    pub const N: Self = Self::NEGATIVE;
    pub const Z: Self = Self::ZERO;
    pub const C: Self = Self::CARRY;
    pub const V: Self = Self::OVERFLOW;

    pub const NZCV_MASK: Self = Self(Self::N.0 | Self::Z.0 | Self::C.0 | Self::V.0);
}

#[derive(bytemuck::Zeroable, Clone)]
// use `repr(C)` so that we can put hot field s next to each other
// so they land on the same cacheline and so that hot fields have
// smaller constant indices to fit inline in an instruction encoding
// rather than an integer immediate, but do note that repr(C) is NOT
// required for safety, and all offset calculations must use `offset_of!`
// this is only here as an optimization and not for correctness
// that is why, we target `repr(Rust)` on debug and miri builds
// to catch any bugs caused by not using `offset_of!`
// we exclude doc so rustdoc doesn't advertise this as a public layout guarantee
#[cfg_attr(not(any(doc, debug_assertions, miri)), repr(C))]
pub struct ExecState {
    pub pc: u64,
    pub x_registers: [u64; X_REGISTER_COUNT as usize],
    pub sp: u64,
    pub pstate: PState,
    pub fpsr: u32,
    pub fpcr: u32,
    pub vectors: [Vector; 32],
}

impl ExecState {
    #[inline(always)]
    pub const fn initial() -> Self {
        bytemuck::zeroed()
    }
}
