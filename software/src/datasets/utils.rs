use crate::random::distribution::Uniform;
use crate::random::generator::Generator;

pub fn shuffle<RNG, const S: usize, const F: usize>(
    mut data: [[f32; F]; S],
    mut labels: [f32; S],
    rng: &mut RNG
) -> ([[f32; F]; S], [f32; S])
    where
        RNG: Generator<u32>
{
    for i in 0..S {
        let j = libm::roundf(
            Uniform::single(0.0, i as f32, rng).unwrap()
        ) as usize;

        data.swap(i, j);
        labels.swap(i, j);
    }

    (data, labels)
}