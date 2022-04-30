pub struct List {
    head: Link,
}

enum Link {
    Empty,
    More(Box<Node>),
}

struct Node {
    elem: i32,
    next: Link,
}

impl List {
    pub fn new() -> Self {
        List { head: Link::Empty }
    }
}

impl List {
    pub fn push(&mut self, elem: i32) {
        let new_node = Node {
            elem: elem,
            next: self.head.clone(),
        };
    }
}

impl Clone for Link {
    fn clone(&self) -> Link {
        match self {
            Link::Empty => Link::Empty,
            Link::More(next) => Link::More(next.clone()),
        }
    }
}

impl Clone for Node {
    fn clone(&self) -> Node {
        Node {
            elem: self.elem,
            next: self.next.clone(),
        }
    }
}