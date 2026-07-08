use core::ops::{Add, Sub, Mul, Div};

pub type Vector<const N: usize> = BaseArray<f32, N, 1>;
pub type Array<const N: usize, const M: usize> = BaseArray<f32, N, M>;

#[derive(Copy, Clone)]
pub struct BaseArray<TElement, const N: usize, const M: usize> {
    shape: (usize, usize),
    layout: [[TElement; M]; N]
}

impl<const N: usize, const M: usize> BaseArray<f32, N, M> {

    pub fn zeros() -> BaseArray<f32, N, M> {
        BaseArray {
            shape: (N, M),
            layout: [[0.0; M]; N]
        }
    }

    pub fn ones() -> BaseArray<f32, N, M> {
        BaseArray {
            shape: (N, M),
            layout: [[1.0; M]; N]
        }
    }

    pub fn full(value: f32) -> BaseArray<f32, N, M> {
        BaseArray {
            shape: (N, M),
            layout: [[value; M]; N]
        }
    }

    pub fn consecutive() -> Vector<N> {
        Vector {
            shape: (N, 1),
            layout: core::array::from_fn(|i| [i as f32])
        }
    }

    pub fn array(self) -> [[f32; M]; N] {
        self.layout
    }
}

impl<TElement, const N: usize, const M: usize> Add<BaseArray<TElement, N, M>> for BaseArray<TElement, N, M>
    where
        TElement: Add<Output = TElement> + Copy
{
    type Output = BaseArray<TElement, N, M>;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            for j in 0..M {
                self.layout[i][j] = self.layout[i][j] + rhs.layout[i][j]
            }
        }

        self
    }
}

impl<TElement, const N: usize, const M: usize> Add<TElement> for BaseArray<TElement, N, M>
    where
        TElement: Add<Output = TElement> + Copy
{
    type Output = BaseArray<TElement, N, M>;

    fn add(mut self, rhs: TElement) -> Self::Output {
        for i in 0..N {
            for j in 0..M {
                self.layout[i][j] = rhs + self.layout[i][j]
            }
        }

        self
    }
}

impl<const N: usize, const M: usize> Add<Array<N, M>> for f32 {
    type Output = Array<N, M>;

    fn add(self, rhs: Array<N, M>) -> Self::Output {
        rhs + self
    }
}

impl<TElement, const N: usize, const M: usize> Mul<BaseArray<TElement, N, M>> for BaseArray<TElement, N, M>
    where
        TElement: Mul<Output = TElement> + Copy
{
    type Output = BaseArray<TElement, N, M>;

    fn mul(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            for j in 0..M {
                self.layout[i][j] = self.layout[i][j] * rhs.layout[i][j]
            }
        }

        self
    }
}

impl<TElement, const N: usize, const M: usize> Mul<TElement> for BaseArray<TElement, N, M>
    where
        TElement: Mul<Output = TElement> + Copy
{
    type Output = BaseArray<TElement, N, M>;

    fn mul(mut self, rhs: TElement) -> Self::Output {
        for i in 0..N {
            for j in 0..M {
                self.layout[i][j] = rhs * self.layout[i][j]
            }
        }

        self
    }
}

impl<const N: usize, const M: usize> Mul<Array<N, M>> for f32 {
    type Output = Array<N, M>;

    fn mul(self, rhs: Array<N, M>) -> Self::Output {
        rhs * self
    }
}

impl<TElement, const N: usize> BaseArray<TElement, N, 1>
    where
        TElement: Copy
{
    pub fn squeeze(self) -> [TElement; N] {
        self.layout.map(|element| element[0])
    }
}

impl<TElement, const N: usize> From<[TElement; N]> for BaseArray<TElement, N, 1> {
    fn from(value: [TElement; N]) -> BaseArray<TElement, N, 1> {
        BaseArray {
            shape: (N, 1),
            layout: value.map(|element| [element])
        }
    }
}

impl<TElement, const N: usize, const M: usize> From<[[TElement; M]; N]> for BaseArray<TElement, N, M> {
    fn from(value: [[TElement; M]; N]) -> Self {
        BaseArray {
            shape: (N, M),
            layout: value
        }
    }
}

impl<TElement, const N: usize, const M: usize> From<&[[TElement; M]; N]> for BaseArray<TElement, N, M>
    where
        TElement: Copy
{
    fn from(value: &[[TElement; M]; N]) -> Self {
        BaseArray {
            shape: (N, M),
            layout: *value
        }
    }
}

pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(&self, rhs: Rhs) -> Self::Output;
}

impl<const N: usize> Dot for [f32; N] {
    type Output = f32;

    fn dot(&self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
}

impl<const N: usize> Dot<&Self> for [f32; N] {
    type Output = f32;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
}

impl<const N: usize> Dot<[f32; N]> for &[f32; N] {
    type Output = f32;

    fn dot(&self, rhs: [f32; N]) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
}

impl<const N: usize> Dot<&Self> for &[f32; N] {
    type Output = f32;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
}

impl<const N: usize> Dot<BaseArray<f32, N, 1>> for BaseArray<f32, N, 1> {
    type Output = f32;

    fn dot(&self, rhs: BaseArray<f32, N, 1>) -> Self::Output {
        self.squeeze().dot(rhs.squeeze())
    }
}

impl<const N: usize> Dot<&BaseArray<f32, N, 1>> for BaseArray<f32, N, 1> {
    type Output = f32;

    fn dot(&self, rhs: &BaseArray<f32, N, 1>) -> Self::Output {
        self.squeeze().dot(rhs.squeeze())
    }
}

impl<const N: usize> Dot<&mut BaseArray<f32, N, 1>> for BaseArray<f32, N, 1> {
    type Output = f32;

    fn dot(&self, rhs: &mut BaseArray<f32, N, 1>) -> Self::Output {
        self.squeeze().dot(rhs.squeeze())
    }
}