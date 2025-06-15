package com.karuneshpalekar.netraa.utils


class LuminosityQueue : QueueL<Long, LuminosityState> {

    private val queue: ArrayDeque<Long> = ArrayDeque()
    private var abovePar: Long = 0
    private var belowPar: Long = 0

    override fun enqueue(element: Long) {
        if (isEmpty) {
            abovePar = element + 25
            //It is either 0 / positive
            belowPar = if (element - 25 > 0) {
                element - 25
            } else {
                0
            }
        }
        if (element in abovePar.downTo(belowPar)) {
            queue.addFirst(element)
        } else {
            abovePar = 0
            belowPar = 0
            queue.clear()
        }
    }

    override fun dequeue() {
        if (isEmpty)
            return
        queue.removeAt(count - 1)
    }

    override val count: Int
        get() = queue.count()

    override fun average(): LuminosityState {
        var sum = 0L
        val latest = queue.take(MAX_TO_BE_CONSIDERED)
        latest.forEach { value ->
            sum += value
        }
        if (sum == 0L || queue.count() == 0) {
            return LuminosityState.UnknownState
        }
        val avg = sum / queue.count()
        return when {
            avg < 1L -> {
                LuminosityState.UnknownState
            }
            avg in 1L..120L -> {
                LuminosityState.LessLight
            }
            avg in 121L..125L -> {
                LuminosityState.AdequateLight
            }
            else -> {
                LuminosityState.ExcessLight
            }
        }
    }

    override fun peek(): Long? =
        queue.getOrNull(0)


    override fun clear() =
        queue.clear()


    override val isEmpty: Boolean
        get() = super.isEmpty


    companion object {
        const val MAX_TO_BE_CONSIDERED = 10
    }

}