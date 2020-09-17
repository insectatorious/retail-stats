import numpy as np
from pytest import approx
from pytest import raises

from retail_stats.elasticity import calculate_cross_elasticity
from retail_stats.elasticity import get_all_cross_elasticities


def __test_bulk_cross_elasticity(original_quantity,
                                 new_quantity,
                                 original_price,
                                 new_price,
                                 expected_ced,
                                 repetition_count):

  original_quantity = np.repeat(original_quantity, repetition_count)
  new_quantity = np.repeat(new_quantity, repetition_count)
  original_price = np.repeat(original_price, repetition_count)
  new_price = np.repeat(new_price, repetition_count)

  expected_ced = np.repeat(expected_ced, repetition_count)
  ced = calculate_cross_elasticity(original_quantity, new_quantity, original_price, new_price)

  assert expected_ced == approx(ced)


def test_substitute_cross_elasticity():
  original_quantity = 200
  new_quantity = 400

  original_price = 1000
  new_price = 1050
  # (200 / 300) / (50 / 1025)
  expected_ced = 13.66666666666666
  ced = calculate_cross_elasticity(original_quantity, new_quantity, original_price, new_price)

  assert expected_ced == approx(ced)

  __test_bulk_cross_elasticity(original_quantity,
                               new_quantity,
                               original_price,
                               new_price,
                               expected_ced,
                               100)


def test_complimentary_cross_elasticity():
  original_quantity = 1000
  new_quantity = 1100

  original_price = 100
  new_price = 80

  expected_ced = -0.4285714286
  ced = calculate_cross_elasticity(original_quantity, new_quantity, original_price, new_price)

  assert expected_ced == approx(ced)

  __test_bulk_cross_elasticity(original_quantity,
                               new_quantity,
                               original_price,
                               new_price,
                               expected_ced,
                               100)


def test_unrelated_cross_elasticity():
  original_quantity = new_quantity = 1000

  original_price = 20
  new_price = 30

  expected_ced = 0
  ced = calculate_cross_elasticity(original_quantity, new_quantity, original_price, new_price)

  assert expected_ced == approx(ced)

  __test_bulk_cross_elasticity(original_quantity,
                               new_quantity,
                               original_price,
                               new_price,
                               expected_ced,
                               100)


def test_mismatched_arrays():
  with raises(ValueError):
    calculate_cross_elasticity(np.random.random(100),
                               np.random.random(10),
                               np.random.random(100),
                               np.random.random(100))

  with raises(ValueError):
    calculate_cross_elasticity(np.random.random(100),
                               np.random.random(100),
                               np.random.random(10),
                               np.random.random(100))

  with raises(ValueError):
    calculate_cross_elasticity(np.random.random(100),
                               np.random.random(100),
                               np.random.random(100),
                               np.random.random(10))

  with raises(ValueError):
    calculate_cross_elasticity(np.random.random(10),
                               np.random.random(1001),
                               np.random.random(103),
                               np.random.random(9))


def test_all_cross_elasticities():
  skus = list("ABCD")
  # [original, new]
  qty_a = [200, 0]
  qty_b = [200, 400]
  prc_a = [1000, 1050]
  prc_b = [1000, 1000]

  qty_c = [1000, 1050]
  qty_d = [1000, 1100]
  prc_c = [100, 80]
  prc_d = [80, 80]

  original_quantities = [qty_a[0], qty_b[0], qty_c[0], qty_d[0]]
  new_quantities = [qty_a[1], qty_b[1], qty_c[1], qty_d[1]]
  original_prices = [prc_a[0], prc_b[0], prc_c[0], prc_d[0]]
  new_prices = [prc_a[1], prc_b[1], prc_c[1], prc_d[1]]

  """
  Cross Elasticities between pairs A,B and C,D
  
    | A | B | C | D 
  A |   |   |   |
  B |   |   |   | 
  C |   |   |   | 
  D |   |   |   |
  """

  ceds = get_all_cross_elasticities(original_quantities=original_quantities,
                                    new_quantities=new_quantities,
                                    original_prices=original_prices,
                                    new_prices=new_prices)

  assert ceds.shape == (len(skus), len(skus))
  assert ceds[(skus.index("A"), skus.index("A"))] == approx(-41)
  assert ceds[(skus.index("B"), skus.index("A"))] == approx(13.66666666666666)
  assert ceds[(skus.index("D"), skus.index("C"))] == approx(-0.4285714286)
  assert ceds[(skus.index("C"), skus.index("A"))] == approx(1)
  assert ceds[(skus.index("A"), skus.index("C"))] == approx(9)
