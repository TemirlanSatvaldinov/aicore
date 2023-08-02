pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub mod sgd;
pub mod function;
pub mod matrix;
pub mod cnn;
mod simd; 
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
