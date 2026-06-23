use std::mem::MaybeUninit;

/// # Safety
///
/// A type that can initialize itself directly inside caller-provided storage.
///
/// `InitInPlace::init` writes a valid `Self` into the provided `MaybeUninit<Self>`
/// and returns a mutable reference to the initialized value.
pub unsafe trait InitInPlace: Sized {
    /// Initializes `this` in place and returns a mutable reference to the
    /// initialized value.
    ///
    /// After this function returns normally, the memory referenced by `this`
    /// must contain a fully initialized, valid `Self`.
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self;
}

unsafe impl<T: Default> InitInPlace for T {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        this.write(T::default())
    }
}
