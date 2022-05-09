/*
如何实现？这里有三个主要的数据结构：
（1）BinaryHeap用于实现 min Heap,此heap需要存储Reverse<节点>
（2）HashSet，实现OpenSet，CloseSet快速查找
（3）二维数组（存储实际的地图）
  close set中需要存储什么？个人认为存储的应该是一个三元组(i32, i32, i32) 最后一个值是其父亲的Hash值
  open set与close set类型一样：HaspMap<i32, Pos3>
  min heap (open set 带分数) 则是 BinaryHeap<Node>，注意Node中存储启发分数的负值（以构建最小堆）
*/
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use array2d::Array2D;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Pos2(pub i32, pub i32);
impl Pos2 {
    pub fn new(x: i32, y: i32) -> Pos2 {
        Pos2(x, y)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Pos3(pub i32, pub i32, pub i32);
impl Pos3 {
    pub fn hash(x: i32, y: i32) -> i32 {
        y << 16 + x
    }

    pub fn hash(&self) -> i32 {
        Pos3::hash(self.0, self.1)
    }

}

// ============= Node for min heap ===============
#[derive(Eq)]
pub struct Node {
    pub score: i32,
    pub Pos3: Pos3
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

pub struct Astar<'a> {
    open_set: HashMap<i32, Pos3>,
    close_set: HashMap<i32, Pos3>,
    heap: BinaryHeap<Node>,
    current_pos: Pos3,
    start_pos: Pos2,
    goal_pos: Pos2,
    map: &'a Array2D<i32>
}

impl<'a> Astar<'a> {
    pub fn new(start_p: &Pos2, end_p: &Pos2, map_ref: &'a Array2D<i32>) -> Astar<'a> {
        let astar_ret = Astar {
            open_set: HashMap::new(),
            close_set: HashMap::new(),
            heap: BinaryHeap::new(),
            current_pos: Pos3(start_p.0, start_p.1, -1),
            start_pos: Pos2(start_p.0, start_p.1),
            goal_pos: Pos2(end_p.0, end_p.1),
            map: map_ref
        };
        astar_ret
    }

    pub fn score(&self, x: i32, y: i32) -> i32 {
        0
    }

    pub fn neighbor_search(&mut self) {
        let current_hash = Pos3::hash(current_pos.0, current_pos.1);
        for dx in (-1)..2 {
            for dy in (-1)..2 {
                if (dx == 0) && (dy == 0) {continue;}                   // 当前点，跳出
                let x = current_pos.0 + dx;
                let y = current_pos.1 + dy;
                let hash_xy = Pos3::hash(x, y);
                if close_set.contains_key(&hash_xy) {continue;}         // 当前点在 close set中
                if open_set.contains_key(&hash_xy) {continue;}          // 当前点在 open set中 直接跳过
                let new_node = Node {
                    score: self.score(x, y),
                    Pos3(x, y, current_hash)
                }
                open_set.insert(hash_xy, );
            }
        }
    }
}

mod test {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[test]
    fn min_heap_test() {
        let mut min_heap: BinaryHeap<Reverse<super::Node>> = BinaryHeap::new();
        let n1 = super::Node{score: 21, Pos2: super::Pos3(4, 5, 10)};
        let n2 = super::Node{score: 4, Pos2: super::Pos3(1, 2, 23)};
        let n3 = super::Node{score: 78, Pos2: super::Pos3(6, 8, 9)};
        let n4 = super::Node{score: 5, Pos2: super::Pos3(9, 3, -3)};
        min_heap.push(Reverse(n1));
        min_heap.push(Reverse(n2));
        min_heap.push(Reverse(n3));
        min_heap.push(Reverse(n4));
        let result = [4, 5, 21, 78];
        let mut cnt = 0;
        while let Some(node) = min_heap.pop() {
            assert_eq!(node.0.score, result[cnt]);
            cnt += 1;
        }
    }
}