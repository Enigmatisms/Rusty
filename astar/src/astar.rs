use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, BinaryHeap};
use array2d::Array2D;

static neighbor_table: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Pos2(pub i32, pub i32);

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Pos3(pub i32, pub i32, pub i32);
impl Pos3 {
    pub fn hash(x: i32, y: i32) -> i32 {
        (y << 8) + x
    }
}

// ============= Node for min heap ===============
#[derive(Eq)]
pub struct Node {
    pub start_dist: i32,
    pub score: i32,
    pub pos: Pos3
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
    open_set: HashSet<i32>,
    close_set: HashMap<i32, Pos3>,
    heap: BinaryHeap<Node>,
    start_pos: Pos2,
    goal_pos: Pos2,
    map: &'a mut Array2D<i32>
}

impl<'a> Astar<'a> {
    pub fn new(start_p: &Pos2, end_p: &Pos2, map_ref: &'a mut Array2D<i32>) -> Astar<'a> {
        let astar_ret = Astar {
            open_set: HashSet::new(),
            close_set: HashMap::new(),
            heap: BinaryHeap::new(),
            start_pos: Pos2(start_p.0, start_p.1),
            goal_pos: Pos2(end_p.0, end_p.1),
            map: map_ref
        };
        astar_ret
    }

    #[inline(always)]
    fn score(&self, x: i32, y: i32, start_dist: i32) -> i32 {
        -((self.goal_pos.0 - x).abs() + (self.goal_pos.1 - y).abs() + start_dist)
    }

    fn neighbor_search(&mut self, current_pos: &Node, current_hash: i32) {
        for (dx, dy) in neighbor_table.iter() {
            let x = current_pos.pos.0 + dx;
            let y = current_pos.pos.1 + dy;
            let hash_xy = Pos3::hash(x, y);
            if self.map[(y as usize, x as usize)] > 0 {continue;}           // 是墙，跳过
            if self.open_set.contains(&hash_xy) {continue;}                 // 当前点在 open set中 直接跳过
            let current_dist = current_pos.start_dist + 1;
            self.open_set.insert(hash_xy);
            self.heap.push(Node {
                start_dist: current_dist,
                score: self.score(x, y, current_dist),
                pos: Pos3(x, y, current_hash)
            });
        }
    }

    fn intialize(&mut self) {
        let new_node = Node {
            start_dist: 0,
            score: self.score(self.start_pos.0, self.start_pos.1, 0),
            pos: Pos3(self.start_pos.0, self.start_pos.1, -1)
        };
        self.heap.push(new_node);
        self.open_set.insert(-1);
        self.open_set.insert(Pos3::hash(self.start_pos.0, self.start_pos.1));
    }

    fn trace_back(&mut self, father_hash: i32) {
        let mut hash_ptr = father_hash;
        let mut now_pos: Pos2 = self.goal_pos;
        // for hash_value in self.open_set.iter() {
        //     if *hash_value < 0 {continue;}
        //     let y = hash_value >> 8;
        //     let x = hash_value % 256;
        //     self.map[(y as usize, x as usize)] = 2;
        // }
        // for (_, node) in self.close_set.iter() {
        //     self.map[(node.1 as usize, node.0 as usize)] = 3;
        // }
        while hash_ptr >= 0 {
            self.map[(now_pos.1 as usize, now_pos.0 as usize)] = 4;
            if let Some(prev_pos) = self.close_set.get(&hash_ptr) {
                now_pos = Pos2(prev_pos.0, prev_pos.1);
                hash_ptr = prev_pos.2;
            } else {    
                panic!("There is a orpan node, sad.");
            }
        }
        self.map[(now_pos.1 as usize, now_pos.0 as usize)] = 4;
    }

    pub fn astar_solver(&mut self) {
        self.intialize();
        // 可能需要loop + if else结构
        loop {
            if let Some(current_node) = self.heap.pop() {
                // println!("current node: {}, {}, {}", current_node.pos.0, current_node.pos.1, self.heap.len());
                if current_node.pos.0 == self.goal_pos.0 && current_node.pos.1 == self.goal_pos.1 {
                    self.trace_back(current_node.pos.2);
                    // println!("A* algorithm completed successfully.");
                    break;
                }
                let current_hash = Pos3::hash(current_node.pos.0, current_node.pos.1);
                self.close_set.insert(current_hash, current_node.pos);      // TODO: 考虑重构clone
                self.neighbor_search(&current_node, current_hash);
            } else {
                println!("There is no path from starting point to the goal. {}", self.heap.len());
                break;
            }
        }
    }
}

mod test {
    
    #[test]
    fn min_heap_test() {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut min_heap: BinaryHeap<Reverse<super::Node>> = BinaryHeap::new();
        let n1 = super::Node{score: 21, start_dist: 0, pos: super::Pos3(4, 5, 10)};
        let n2 = super::Node{score: 4, start_dist: 0, pos: super::Pos3(1, 2, 23)};
        let n3 = super::Node{score: 78, start_dist: 0, pos: super::Pos3(6, 8, 9)};
        let n4 = super::Node{score: 5, start_dist: 0, pos: super::Pos3(9, 3, -3)};
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