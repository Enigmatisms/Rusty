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
    pub fn new() -> MultiLink<T> {
        MultiLink{head: None}
    }
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
        MultiLink{
            head: self.head.as_ref().map(|node|{
                node.next.clone()
            }).unwrap()
        }
    }
}

// https://stackoverflow.com/questions/53229116/what-does-it-mean-when-a-struct-has-two-lifetime-parameters#:~:text=What%20do%20the%20lifetime%20specifiers,the%20lifetime%20of%20that%20member.
// an instance of that struct can't outlive the lifetime of that member （也即，Iter的存活时间（也即引用变量this的存活时间）至多与指向的node一样长（再长一些就可能指向失效数据））
pub struct Iter<'a, T> {
    this: Option<&'a Rc<Node<T>>>               // 表示此引用至少要与Iter的存活时间一样
}

impl<T> MultiLink<T> {
    pub fn iter(&self) -> Iter<T> {
        Iter{
            this: self.head.as_ref()
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {          // 告知编译器，在rc_node失效时，此指针也失效(this指向的内容失效之前，Iter就应该失效)
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.this.map(|rc_node|{
            self.this = rc_node.next.as_ref();
            &rc_node.elem
        })
    }
}

impl<T> Drop for MultiLink<T> {
    fn drop(&mut self) {
        let mut ptr = self.head.take();         // take 对应了move，因为take返回Option<T>
        // 为什么这里可以take呢？因为链表中一定没有关于self.head的引用（有的话就不是head了）
        while let Some(rc) = ptr {            // 此node是Rc<Node<T>>
            match Rc::try_unwrap(rc) {
                Ok(mut node) => {               // 此处的mut与ref一样，都是node类型的一个标识
                    ptr = node.next.take();
                },
                _ => {
                    break;
                }
            }
            // if let Ok(mut node) = Rc::try_unwrap(rc) {      // node就是 Node<T>
            //     ptr = node.next.take();
            // } else {
            //     break;
            // }
        }
    }
}

