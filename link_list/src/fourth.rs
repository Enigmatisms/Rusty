// 双向链表
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Display;

pub struct DoubleList<T> {
    head: Link<T>,
    tail: Link<T>
}

type Link<T> = Option<Rc<RefCell<Node<T>>>>;

struct Node<T> {
    elem: T,
    prev: Link<T>,
    next: Link<T>
}

impl<T> DoubleList<T> {
    pub fn new() -> DoubleList<T> {
        DoubleList{
            head: None,
            tail: None
        }
    }
}

impl<T> DoubleList<T> {
    pub fn push_front(&mut self, elem: T) {
        // 新的Node需要设置head与tail
        let new_node = Node {
            elem: elem,
            prev: None,
            next: self.head.clone()
        };
        let wrapped_node = Rc::new(RefCell::new(new_node));
        if let Some(node) = self.head.take() {
            let mut inner_node = node.borrow_mut();
            // let _:() = inner_node;
            inner_node.prev = Some(wrapped_node.clone());           // Rc的clone应该开销比较小
            self.head = Some(wrapped_node);
        } else {                                        // 如果原来链表为空，则需要设置tail
            self.tail = Some(wrapped_node.clone());
            self.head = Some(wrapped_node);
        }
    }

    pub fn push_back(&mut self, elem: T) {
        let new_node = Node {
            elem: elem,
            prev: self.tail.clone(),
            next: None
        };
        let wrapped_node = Some(Rc::new(RefCell::new(new_node)));
        if let Some(node) = self.tail.take() {
            let mut inner_node = node.borrow_mut();
            // let _:() = inner_node;
            inner_node.next = wrapped_node.clone();           // Rc的clone应该开销比较小
            self.tail = wrapped_node;
        } else {                                        // 如果原来链表为空，则需要设置tail
            self.tail = wrapped_node.clone();
            self.head = wrapped_node;
        }
    }
    

    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node|{            // node 类型是 Rc<RefCell<Node<T>>>
            if let Some(next_node) = node.borrow_mut().next.take() {
                next_node.borrow_mut().prev.take();         // 消除一个ref_count
                self.head = Some(next_node);
            } else {                                        // 只有head（tail）
                self.tail.take();
            }
            Rc::try_unwrap(node).ok().unwrap().into_inner().elem
        })
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.take().map(|node|{
            if let Some(prev_node) = node.borrow_mut().prev.take() {
                prev_node.borrow_mut().next.take();
                self.tail = Some(prev_node);
            } else {
                self.head.take();
            }
            Rc::try_unwrap(node).ok().unwrap().into_inner().elem
        })
    }
}

impl<T: Display> DoubleList<T> {
    pub fn show_links(&self) {
        let mut ptr = self.head.clone();
        while let Some(rc) = ptr {
            ptr = rc.borrow().next.clone();
            print!("{}, ", rc.borrow().elem);
        }
        print!("\n");
    }
}

impl<T> Drop for DoubleList<T> {
    // 显然还有更简单的方法，直接复用pop
    fn drop(&mut self) {
        let mut ptr = self.head.take();
        while let Some(node) = ptr {
            node.borrow_mut().prev.take();
            ptr = node.borrow_mut().next.take();
        }
        self.tail.take();
    }
}

mod test{
    #[test]
    fn pop_back_test() {
        let mut dl = super::DoubleList::new();
        dl.push_back(4);
        dl.push_back(5);
        dl.push_back(6);
        dl.push_front(1);
        dl.push_front(2);
        assert_eq!(dl.pop_back(), Some(6));
        assert_eq!(dl.pop_back(), Some(5));
        assert_eq!(dl.pop_back(), Some(4));
        assert_eq!(dl.pop_back(), Some(1));
        assert_eq!(dl.pop_back(), Some(2));
        assert_eq!(dl.pop_back(), None);
    }

    #[test]
    fn pop_front_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_back(4);
        dl.push_back(5);
        dl.push_back(6);
        assert_eq!(dl.pop_front(), Some(2));
        assert_eq!(dl.pop_front(), Some(1));
        assert_eq!(dl.pop_front(), Some(4));
        assert_eq!(dl.pop_front(), Some(5));
        assert_eq!(dl.pop_front(), Some(6));
        assert_eq!(dl.pop_back(), None);
    }

    #[test]
    fn casual_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_back(4);
        assert_eq!(dl.pop_front(), Some(1));
        assert_eq!(dl.pop_front(), Some(4));
        dl.push_back(5);
        dl.push_front(6);
        dl.push_front(7);
        dl.push_front(8);
        assert_eq!(dl.pop_front(), Some(8));
        assert_eq!(dl.pop_back(), Some(5));
        assert_eq!(dl.pop_front(), Some(7));
        assert_eq!(dl.pop_back(), Some(6));
        assert_eq!(dl.pop_front(), None);
        assert_eq!(dl.pop_back(), None);
    }
}