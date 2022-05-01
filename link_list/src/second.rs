use std::mem;
use std::fmt::Display;

pub struct List<T> where T: Default {
    head: Link<T>,
}

enum Link<T> where T: Default {
    Empty,
    More(Box<Node<T>>),
}

struct Node<T> where T: Default {
    elem: T,
    next: Link<T>,
}

impl<T: Default> List<T> {
    pub fn new() -> Self {
        List { head: Link::Empty }
    }
}

impl<T: Default> List<T> {
    pub fn push(&mut self, elem: T) {
        let new_node: Node<T> = Node {
            elem: elem,
            next: mem::replace(&mut self.head, Link::Empty)
        };
        self.head = Link::More(Box::new(new_node));
    }

    pub fn pop(&mut self) -> T {
        if let Link::More(mut node) = mem::replace(&mut self.head, Link::Empty) {
            let pop_elem = mem::replace(&mut node.elem, T::default());
            self.head = node.next;
            return pop_elem;
        } else {
            panic!("Can not pop from an empty stack.");
        }
    }
}

impl<T: std::fmt::Display + Default> List<T> {
    pub fn show_stack(&self, verbose:bool, single_line:bool) {
        if verbose {
            println!("Current stack:");
        }
        let mut ptr: &Link<T> = &self.head;
        while let Link::More(node) = ptr {
            if single_line {
                print!("{}, ", node.elem);
            } else {
                println!("Stack: {}", node.elem);
            }
            ptr = &node.next;
        }
        if single_line {
            println!("");
        }
    }
}

impl<T: Default> Drop for List<T> {
    fn drop(&mut self) {
        let mut ptr = mem::replace(&mut self.head, Link::Empty);
        while let Link::More(mut node) = ptr{
            ptr = mem::replace(&mut node.next, Link::Empty);
        }
    }
}

mod test {
    #[test]
    fn basics() {
        let mut l1: super::List<i32> = super::List::new(); 
        
        l1.push(4);
        l1.push(6);
        l1.push(1);
        assert_eq!(l1.pop(), 1);
        assert_eq!(l1.pop(), 6);
        assert_eq!(l1.pop(), 4);
        l1.push(3);
        l1.push(2);
        l1.push(5);
        assert_eq!(l1.pop(), 5);
        assert_eq!(l1.pop(), 2);
        l1.push(8);
        assert_eq!(l1.pop(), 8);
        assert_eq!(l1.pop(), 3);
    }
}