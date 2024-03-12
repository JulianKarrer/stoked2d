
/// Efficiently calculates the next multiple of 'multiple_of' which is greater or
/// equal to 'n'. Equivalent to the rounded up result of division.
/// 
/// As seen on [StackOverflow](https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c)
pub fn ceil_div(n:usize, multiple_of:usize)->usize{
  (1 + ((n - 1) / multiple_of)) * multiple_of
}