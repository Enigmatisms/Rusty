use std::cmp::Ordering;
use array2d::Array2D;

#[derive(Eq)]
pub struct Test {
    pub score: i32,
    pub name: String,
    pub id: i32
}

impl Ord for Test {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for Test {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Test {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

pub struct Pos3(pub i32, pub i32, pub i32);

impl Pos3 {
    pub fn hash(&self) -> i32 {
        if self.0 >= self.1 {
            self.0 * self.0 + self.1
        } else {
            self.1 * self.1 + self.0
        }
    }
}
