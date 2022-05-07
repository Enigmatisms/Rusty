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
        let wrapped_node = Some(Rc::new(RefCell::new(new_node)));
        if let Some(node) = self.head.take() {
            let mut inner_node = node.borrow_mut();
            // let _:() = inner_node;
            inner_node.prev = wrapped_node.clone();           // Rc的clone应该开销比较小
            self.head = wrapped_node;
        } else {                                        // 如果原来链表为空，则需要设置tail
            self.tail = wrapped_node.clone();
            self.head = wrapped_node;
        }
    }

    pub fn push_back(&mut self, elem: T) {
        let new_node = Node {
            elem: elem,
            prev: self.head.clone(),
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
        self.head.take().and_then(|node| {
            let mut inner_node = Rc::try_unwrap(node).ok().unwrap().into_inner();
            if let Some(next_node) = inner_node.next.take() {           // take返回值，信息都在next_node中，只需要根据next_node重写head即可，故可以take
                next_node.borrow_mut().prev = None;
                self.head = Some(next_node);
            } else {
                self.tail = None;
            }
            Some(inner_node.elem)
        })
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.take().and_then(|node|{            // 如果tail是None，就不会执行此函数
            let mut inner_node = match Rc::try_unwrap(node).ok() {
                Some(node) => node.into_inner(),
                None => return None
            };
            if let Some(prev_node) = inner_node.next.take() {
                prev_node.borrow_mut().next = None;
                self.tail = Some(prev_node);
            } else {
                self.head = None;
            }
            Some(inner_node.elem)
        }) 
    }   // 注意，map需要返回值是U，而不是Option<U>
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