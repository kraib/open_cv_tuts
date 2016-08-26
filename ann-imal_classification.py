import cv2
import numpy as np
from random import randint

animals_net = cv2.ml.ANN_MLP_create()
animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
animals_net.setLayerSizes(np.array([3, 6, 4]))
animals_net.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))


def dog_sample():
    return [randint(5, 20), 1, randint(38, 42)]


def dog_class():
    return [1, 0, 0, 0]


def condor_sample():
    return [randint(3, 13), 3, 0]


def condor_class():
    return [0, 1, 0, 0]


def dolfin_sample():
    return [randint(20, 190), randint(5, 15), randint(80, 100)]


def dolfin_class():
    return [0, 0, 1, 0]


def dragon_sample():
    return [randint(1200, 1800), randint(15, 40), randint(110, 180)]


def dragon_class():
    return [0, 0, 0, 1]


# SAMPLES = 5000
#
# for x in range(0, SAMPLES):
#     print "Samples %d%d" % (x, SAMPLES)
#     animals_net.train(np.array([dog_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
#                       np.array([dog_class()], dtype=np.float32))
#     animals_net.train(np.array([condor_sample()], dtype=np.float32),
#                       cv2.ml.ROW_SAMPLE, np.array([condor_class()], dtype=np.float32))
#     animals_net.train(np.array([dolfin_sample()], dtype=np.float32),
#                       cv2.ml.ROW_SAMPLE, np.array([dolfin_class()], dtype=np.float32))
#     animals_net.train(np.array([dragon_sample()], dtype=np.float32),
#                       cv2.ml.ROW_SAMPLE, np.array([dragon_class()], dtype=np.float32))
#
# print animals_net.predict(np.array([dog_sample()], dtype=np.float32))
# print animals_net.predict(np.array([condor_sample()], dtype=np.float32))
# print animals_net.predict(np.array([dragon_sample()], dtype=np.float32))
def record(sample, classification):
  return (np.array([sample], dtype=np.float32), np.array([classification], dtype=np.float32))

records = []
RECORDS = 5000
for x in range(0, RECORDS):
  records.append(record(dog_sample(), dog_class()))
  records.append(record(condor_sample(), condor_class()))
  records.append(record(dolfin_sample(), dolfin_class()))
  records.append(record(dragon_sample(), dragon_class()))

EPOCHS = 200
for e in range(0, EPOCHS):
  print "Epoch %d:" % e
  for t, c in records:
    animals_net.train(t, cv2.ml.ROW_SAMPLE, c)


TESTS = 100
dog_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([dog_sample()], dtype=np.float32))[0])
  print "class: %d" % clas
  if (clas) == 0:
    dog_results += 1

condor_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([condor_sample()], dtype=np.float32))[0])
  print "class: %d" % clas
  if (clas) == 1:
    condor_results += 1

dolphin_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([dolfin_sample()], dtype=np.float32))[0])
  print "class: %d" % clas
  if (clas) == 2:
    dolphin_results += 1

dragon_results = 0
for x in range(0, TESTS):
  clas = int(animals_net.predict(np.array([dragon_sample()], dtype=np.float32))[0])
  print "class: %d" % clas
  if (clas) == 3:
    dragon_results += 1

print "Dog accuracy: %f%%" % (dog_results)
print "condor accuracy: %f%%" % (condor_results)
print "dolphin accuracy: %f%%" % (dolphin_results)
print "dragon accuracy: %f%%" % (dragon_results)