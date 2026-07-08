use libm;

use crate::linalg::{Array, BaseArray, Dot, Vector};

use crate::random::distribution::{Gaussian, Uniform};
use crate::random::generator::{Generator, XORShift32};

pub fn unpackbits(value: u8) -> [f32; 8] {
    let mut bits = [0.0; 8];

    for i in 0..8 {
        let mask = 1 << i;

        if (value & mask) != 0 {
            bits[i] = 1.0
        }
    }

    bits
}

pub fn centroids<RNG, const S: usize>(separation: f32, rng: &mut RNG) -> ([[f32; 4]; S], [f32; S])
    where
        RNG: Generator<u32>
{
    let centroids: Array<4, 4> = Array::from(
        core::array::from_fn(
            |i| {
                let mut bits = unpackbits(i as u8);

                [bits[0], bits[1], bits[2], bits[3]]
            }
        )
    );

    let centroids = 2.0 * centroids * separation;
    let centroids = centroids + (-separation);

    let centroids = centroids.array();

    let mut data = core::array::from_fn(
        |_| {
            Gaussian::sample(0.0, 1.0, 100, rng)
                .expect("Not enough samples were generated within the specified limit.")
        }
    );

    let labels = core::array::from_fn(|i| {
        if i > (S / 2) {
            1.0
        } else {
            0.0
        }
    });

    for (centroid, chunk) in data
        .chunks_mut(S / centroids.len())
        .enumerate()
    {
        for features in chunk.iter_mut() {
            features[0] += centroids[centroid][0];
            features[1] += centroids[centroid][1];
            features[2] += centroids[centroid][2];
            features[3] += centroids[centroid][3];
        }
    }

    (data, labels)
}

