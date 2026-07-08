use crate::random::distribution::Uniform;
use crate::random::generator::{Generator, XORShift32};

pub fn redundant<RNG, const S: usize>(rng: &mut RNG) -> ([[f32; 4]; S], [f32; S])
    where
        RNG: Generator<u32>
{
    let data = core::array::from_fn(
        |_| { Uniform::sample(0.0, 1.0, rng).unwrap() }
    );

    let labels = data.map(
        |element| if element[0] + element[2] > 1.0 { 1.0 } else { 0.0 }
    );

    (data, labels)
}

pub fn dot<TElement, const N: usize>(lhs: &[TElement; N], rhs: &[TElement; N]) -> TElement
    where
        TElement: core::iter::Sum,
        for <'a> &'a TElement: core::ops::Mul<&'a TElement, Output = TElement>
{
    lhs.iter().zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum::<TElement>()
}

fn predict(weights: &[f32; 4], inputs: &[f32; 4]) -> f32 {
    let prediction = [dot(&weights, inputs)]
        .iter()
        .map(|x| x.max(0.0))
        .map(|x| if x > 1.0 { 1.0 } else { 0.0 })
        .sum::<f32>();

    prediction
}

