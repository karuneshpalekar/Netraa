package com.karuneshpalekar.netraa.utils

import android.graphics.RectF
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category

class DetectionQueue(var clasifier: String) : Queue<Element, Long> {
    private val queue: ArrayDeque<Element> = ArrayDeque()

    override fun enqueue(element: Element) {
        if (clasifier == Constants.HUMAN_ANIMAL) {
            if (element.score in MAX_THRESHOLD.downTo(HUMAN_ANIMAL_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else if (clasifier == Constants.FURNITURE) {
            if (element.score in MAX_THRESHOLD.downTo(FURNITURE_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else if (clasifier == Constants.BRG) {
            if (element.score in MAX_THRESHOLD.downTo(BGR_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else if (clasifier == Constants.HUMAN_CLASSIFIER) {
            if (element.score in MAX_THRESHOLD.downTo(HUMAN_CLASSIFIER_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else if (clasifier == Constants.POSTURE) {
            if (element.score in MAX_THRESHOLD.downTo(POSTURE_CLASSIFIER_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else if (clasifier == Constants.ANIMALS) {
            if (element.score in MAX_THRESHOLD.downTo(ANIMAL_MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        } else {
            if (element.score in MAX_THRESHOLD.downTo(MIN_THRESHOLD)) {
                queue.addFirst(element)
            } else {
                clear()
            }
        }
    }

    override fun dequeue() {
        if (isEmpty)
            return

        if (clasifier == Constants.HUMAN_ANIMAL) {
            if (count >= HUMAN_ANIMAL_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        } else if (clasifier == Constants.FURNITURE) {
            if (count >= FURNITURE_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        } else if (clasifier == Constants.BRG) {
            if (count >= BGR_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        } else if (clasifier == Constants.HUMAN_CLASSIFIER) {
            if (count >= HUMAN_CLASSIFIER_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        } else if (clasifier == Constants.POSTURE) {
            if (count >= POSTURE_CLASSIFIER_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        } else if (clasifier == Constants.ANIMALS) {
            if (count >= ANIMAL_MAX_TO_BE_CONSIDERED) {
                queue.removeAt(count - 1)
            }
        }

    }

    override val count: Int
        get() = queue.count()

    override fun average(): Long {
        var sum = 0L
        queue.forEach { value ->
            sum += value.score
        }
        if (sum == 0L || queue.count() == 0) {
            return 0
        }
        return sum / queue.count()
    }

    override fun peek(): Element? =
        queue.getOrNull(0)

    override fun clear() = queue.clear()

    override val isEmpty: Boolean
        get() = super.isEmpty


    companion object {

        //Human Animal
        const val HUMAN_ANIMAL_MAX_TO_BE_CONSIDERED = 4
        const val HUMAN_ANIMAL_MIN_THRESHOLD = 82

        //BGR
        const val BGR_MAX_TO_BE_CONSIDERED = 5
        const val BGR_MIN_THRESHOLD = 88

        //Furniture
        const val FURNITURE_MAX_TO_BE_CONSIDERED = 6
        const val FURNITURE_MIN_THRESHOLD = 50

        //Animal
        const val ANIMAL_MAX_TO_BE_CONSIDERED = 5
        const val ANIMAL_MIN_THRESHOLD = 82

        //Human Classifier
        const val HUMAN_CLASSIFIER_MIN_THRESHOLD = 60
        const val HUMAN_CLASSIFIER_MAX_TO_BE_CONSIDERED = 4

        //Posture Detection
        const val POSTURE_CLASSIFIER_MIN_THRESHOLD = 80
        const val POSTURE_CLASSIFIER_MAX_TO_BE_CONSIDERED = 5

        //Threshold
        const val MAX_THRESHOLD = 100
        const val MIN_THRESHOLD = 80
    }
}

data class Element(
    var label: String,
    var score: Int,
    var tensorImage: TensorImage,
    var category: Category? = null,
    var boundingBox: RectF? = null
)

