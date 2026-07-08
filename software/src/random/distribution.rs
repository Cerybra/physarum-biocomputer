use libm::{powf, logf};

use core::marker::PhantomData;

use crate::random::generator::Generator;

pub struct Uniform<TGenerator, RNG>
    where
        RNG: Generator<TGenerator>
{
    _phantom_rng: PhantomData<RNG>,
    _phantom_generator: PhantomData<TGenerator>
}

impl<RNG> Uniform<u32, RNG>
    where
        RNG: Generator<u32>
{
    pub fn single(low: f32, high: f32, rng: &mut RNG) -> Result<f32, ()> {
        let sample = rng.sample();
        let sample = f32::from_be_bytes(
            (sample >> 9 | 0x3f800000).to_be_bytes()
        ) - 1.0;

        Ok(sample * (high - low) + low)
    }

    pub fn sample<const S: usize>(
        low: f32,
        high: f32,
        rng: &mut RNG
    ) -> Result<[f32; S], ()> {
        if low < high {
            return Ok(
                core::array::from_fn(
                    |_| { Uniform::single(low, high, rng).unwrap() }
                )
            )
        }

        return Err(())
    }
}

pub struct Exponential<TGenerator, RNG>
    where
        RNG: Generator<TGenerator>
{
    _phantom_rng: PhantomData<RNG>,
    _phantom_generator: PhantomData<TGenerator>
}

impl<RNG> Exponential<u32, RNG>
    where
        RNG: Generator<u32>
{
    pub fn single(rate: f32, rng: &mut RNG) -> Result<f32, ()> {
        Ok(-rate * logf(Uniform::single(0.0, 1.0, rng)?))
    }

    pub fn sample<const S: usize>(rate: f32, rng: &mut RNG) -> [f32; S] {
        core::array::from_fn(
            |_| { Exponential::single(rate, rng).unwrap() }
        )
    }
}

pub struct Gaussian<TGenerator, RNG>
    where
        RNG: Generator<TGenerator>
{
    _phantom_rng: PhantomData<RNG>,
    _phantom_generator: PhantomData<TGenerator>
}

impl<RNG> Gaussian<u32, RNG>
    where
        RNG: Generator<u32>
{
    pub fn sample<const S: usize>(mean: f32, std: f32, limit: usize, rng: &mut RNG) -> Result<[f32; S], ()> {
        let mut index = 0;
        let mut counter = 0;

        let mut samples = [0.0; S];

        while (counter < limit) && (index < S) {
            let (y_1, y_2) = (Exponential::single(1.0, rng)?, Exponential::single(1.0, rng)?);

            if y_2 >= (powf(y_1 - 1.0, 2.0) / 2.0) {
                if Uniform::single(0.0, 1.0, rng)? > 0.5 {
                    samples[index] = (-y_2 * std) + mean;
                } else {
                    samples[index] = (y_2 * std) + mean;
                }

                index += 1;
            }

            counter += 1;
        }

        if index == S {
            return Ok(samples)
        }

        return Err(())
    }
}
