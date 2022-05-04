use std::rc::Rc;

pub struct MultiLink<T> {
    head: Link<T>
}

type Link<T> = Option<Rc<Node<T>>>;

struct Node<T> {
    elem: T,
    next: Link<T>
}

impl<T> MultiLink<T> {
    // 此函数的意义是：返回一个新的链表，此链表以链表原来的head为新链表head的next
    pub fn prepend(&self, new_elem: T) -> MultiLink<T> {
        MultiLink {
            head: Some(Rc::new(Node{
                elem: new_elem,
                next: self.head.clone()
            }))
        }
    }
    
    // 返回一个链表，此链表是当前链表去掉头node
    pub fn tail(&self) -> MultiLink<T> {
        // 直接通过self.head内部的next域访问并且clone应该是不行的
        MultiLink {
            head: match self.head.as_ref() {
                Some(ref node) => {
                    node.next.clone()
                },
                None => {None}
            }

            // self.head.as_ref() 的类型是 Option<&Rc<Node<T>>>
            // map对应的match最好也写一遍免得忘记
            // self.head.as_ref().map(|node| {     // node本身就是Rc::<Node<T>> 其next是Option<...> 而map本身返回一个Option
            //     node.next.clone().unwrap()
            // }).unwrap()
        }
    }
}

