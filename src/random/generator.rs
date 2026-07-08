pub trait Generator<T> {
    fn sample(&mut self) -> T;
}

#[derive(Copy, Clone, Debug)]
pub struct XORShift32(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct XORShift64(pub u64);

impl Generator<u32> for XORShift32 {
    fn sample(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;

        self.0
    }
}

impl Generator<u64> for XORShift64 {
    fn sample(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;

        self.0
    }
}
