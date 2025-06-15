package com.karuneshpalekar.netraa.utils

interface Queue<T, K> {

    fun enqueue(element: T)

    fun dequeue()

    val count: Int

    fun average(): K

    val isEmpty: Boolean
        get() = count == 0

    fun peek(): Element?

    fun clear()
}