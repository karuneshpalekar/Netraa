package com.karuneshpalekar.netraa.utils

interface QueueL<T, K> {

    fun enqueue(element: T)

    fun dequeue()

    val count: Int


    fun average(): K

    val isEmpty: Boolean
        get() = count == 0

    fun peek(): T?

    fun clear()
}