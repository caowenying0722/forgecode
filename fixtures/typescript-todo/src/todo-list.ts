export interface Todo {
  readonly id: number
  readonly title: string
  completed: boolean
}

export class TodoList {
  private readonly todos: Todo[] = []
  private nextId = 1

  add(title: string): Todo {
    const normalizedTitle = title.trim()
    if (!normalizedTitle) {
      throw new Error('title must not be empty')
    }

    const todo: Todo = {
      id: this.nextId++,
      title: normalizedTitle,
      completed: false,
    }
    this.todos.push(todo)
    return { ...todo }
  }

  complete(id: number): Todo {
    const todo = this.todos[id]
    if (!todo) {
      throw new Error('todo not found')
    }

    todo.completed = true
    return { ...todo }
  }

  all(): Todo[] {
    return this.todos.map((todo) => ({ ...todo }))
  }
}
