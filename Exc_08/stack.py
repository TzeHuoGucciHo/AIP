class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

# Example usage:
stack = Stack()
stack.push([1,1])
stack.push([1,2])
stack.push([2,1])
stack.push([2,2])
print("Stack:", stack.stack)
print("Peek:", stack.peek())
print("Pop:", stack.pop())
print("Stack after pop:", stack.stack)
print("Is stack empty?", stack.is_empty())
print("Stack size:", stack.size())


exit()
