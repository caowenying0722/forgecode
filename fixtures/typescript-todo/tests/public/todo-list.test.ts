import { describe, expect, test } from 'vitest'

import { TodoList } from '../../src/todo-list'

describe('TodoList', () => {
  test('creates todos with stable ids', () => {
    const list = new TodoList()

    expect(list.add('write tests')).toEqual({
      id: 1,
      title: 'write tests',
      completed: false,
    })
    expect(list.add('fix implementation').id).toBe(2)
  })

  test('completes the todo selected by id', () => {
    const list = new TodoList()
    const first = list.add('first task')
    list.add('second task')

    expect(list.complete(first.id)).toMatchObject({
      id: first.id,
      completed: true,
    })
    expect(list.all()).toEqual([
      { id: 1, title: 'first task', completed: true },
      { id: 2, title: 'second task', completed: false },
    ])
  })
})
