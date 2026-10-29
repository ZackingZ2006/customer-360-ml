"""Checks for the consequential failure modes of this experiment.

Run: python -m unittest test_temporal_integrity -v
"""
import unittest
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from customer360_ml import FEATURES, build_snapshot, load_data


class TemporalIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers, cls.orders = load_data(Path(__file__).resolve().parent/'data')

    def test_future_order_changes_cannot_change_features(self):
        original = build_snapshot(self.customers, self.orders, '2025-06-30')
        changed = self.orders.copy()
        future = changed.order_date.gt('2025-06-30')
        changed.loc[future, 'order_total_amount'] = 999999999
        changed.loc[future, 'order_status'] = 'Cancelled'
        mutated = build_snapshot(self.customers, changed, '2025-06-30')
        assert_frame_equal(original[['customer_id']+FEATURES], mutated[['customer_id']+FEATURES])

    def test_label_boundaries_and_history_boundary(self):
        customers = pd.DataFrame({'customer_id': ['A','B','C','D'], 'join_date': pd.to_datetime(['2024-01-01']*4)})
        orders = pd.DataFrame([
            ('a0','A','2025-01-01',10), ('a1','A','2025-03-31',20),
            ('b0','B','2025-01-01',10), ('b1','B','2025-06-29',20),
            ('c0','C','2025-01-01',10), ('c1','C','2025-06-30',20),
            ('d0','D','2025-04-01',10),
        ], columns=['order_id','customer_id','order_date','order_total_amount'])
        orders['order_date'] = pd.to_datetime(orders.order_date)
        orders['order_status'] = 'Completed'
        orders['sales_channel'] = 'Store'
        s = build_snapshot(customers, orders, '2025-03-31').set_index('customer_id')
        self.assertEqual(s.repurchase_90d.to_dict(), {'A':0,'B':1,'C':0})
        self.assertEqual(s.loc['A','orders_90d'],2)
        self.assertEqual(s.loc['A','recency_capped_91'],0)
        self.assertNotIn('D',s.index)

    def test_censored_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not fully observed'):
            build_snapshot(self.customers, self.orders, '2025-12-31')
        demo = build_snapshot(self.customers, self.orders, '2025-12-31', labeled=False)
        self.assertNotIn('repurchase_90d', demo.columns)

    def test_cancelled_orders_do_not_count(self):
        original = build_snapshot(self.customers, self.orders, '2025-09-30')
        completed_only = self.orders.loc[self.orders.order_status.eq('Completed')]
        other = build_snapshot(self.customers, completed_only, '2025-09-30')
        assert_frame_equal(original, other)

    def test_one_row_per_known_customer_per_cutoff(self):
        for date in ['2025-03-31','2025-06-30','2025-09-30']:
            frame = build_snapshot(self.customers, self.orders, date)
            self.assertTrue(frame.customer_id.is_unique)
            past_buyers = self.orders.loc[self.orders.order_status.eq('Completed') & self.orders.order_date.le(date), 'customer_id'].nunique()
            self.assertEqual(len(frame), past_buyers)


if __name__ == '__main__':
    unittest.main()
