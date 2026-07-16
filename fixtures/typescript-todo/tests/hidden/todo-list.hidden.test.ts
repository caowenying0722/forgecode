import { describe, expect, test } from 'vitest'

import { TodoList } from '../../src/todo-list'

describe('TodoList hidden cases', () => {
  test('completes a later todo by its id', () => {
    const list = new TodoList()
    list.add('first task')
    const second = list.add('second task')

    expect(list.complete(second.id)).toMatchObject({
      id: second.id,
      completed: true,
    })
  })

  test('rejects an unknown id without changing existing todos', () => {
    const list = new TodoList()
    list.add('keep pending')

    expect(() => list.complete(999)).toThrow('todo not found')
    expect(list.all()[0]?.completed).toBe(false)
  })
})
