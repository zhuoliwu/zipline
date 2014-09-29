#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import random
from itertools import islice, dropwhile, product
import operator
from six.moves import range, map
from nose_parameterized import parameterized
from unittest import TestCase

import pandas as pd
import numpy as np

from zipline.finance.trading import TradingEnvironment
from zipline.utils import events as events_module
from zipline.utils.events import (
    EventRule,
    StatelessRule,
    Always,
    Never,
    InvertedRule,
    AfterOpen,
    ComposedRule,
    BeforeClose,
    OnDate,
    BeforeDate,
    AfterDate,
    AtTime,
    AfterTime,
    BeforeTime,
    HalfDay,
    NotHalfDay,
    NthTradingDayOfWeek,
    NDaysBeforeLastTradingDayOfWeek,
    NthTradingDayOfMonth,
    NDaysBeforeLastTradingDayOfMonth,
    StatefulRule,
    DoNTimes,
    SkipNTimes,
    NTimesPerPeriod,
    RuleFromCallable,
    _build_offset,
    _build_date,
    _build_time,
    EventManager,
    Event,
)


# A day known to be a half day.
HALF_DAY = datetime.date(year=2014, month=7, day=3)

# A day known to be a full day.
FULL_DAY = datetime.date(year=2014, month=9, day=24)


def param_range(*args):
    return ([n] for n in range(*args))


class TestUtils(TestCase):
    @parameterized.expand([
        ('_build_date', _build_date),
        ('_build_time', _build_time),
    ])
    def test_build_none(self, name, f):
        with self.assertRaises(ValueError):
            f(None, {})

    def test_build_offset_both(self):
        with self.assertRaises(ValueError):
            _build_offset(datetime.timedelta(minutes=1), {'minutes': 1})

    def test_build_offset_kwargs(self):
        kwargs = {'minutes': 1}
        self.assertEqual(
            _build_offset(None, kwargs),
            datetime.timedelta(**kwargs),
        )

    def test_build_offset_td(self):
        td = datetime.timedelta(minutes=1)
        self.assertEqual(
            _build_offset(td, {}),
            td,
        )

    def test_build_date_both(self):
        with self.assertRaises(ValueError):
            _build_date(
                datetime.date(year=2014, month=9, day=25), {
                    'year': 2014,
                    'month': 9,
                    'day': 25,
                },
            )

    def test_build_date_kwargs(self):
        kwargs = {'year': 2014, 'month': 9, 'day': 25}
        self.assertEqual(
            _build_date(None, kwargs),
            datetime.date(**kwargs),
        )

    def test_build_date_date(self):
        date = datetime.date(year=2014, month=9, day=25)
        self.assertEqual(
            _build_date(date, {}),
            date,
        )

    def test_build_time_both(self):
        with self.assertRaises(ValueError):
            _build_time(
                datetime.time(hour=1, minute=5), {
                    'hour': 1,
                    'minute': 5,
                },
            )

    def test_build_time_kwargs(self):
        kwargs = {'hour': 1, 'minute': 5}
        self.assertEqual(
            _build_time(None, kwargs),
            datetime.time(**kwargs),
        )


class TestEventManager(TestCase):
    def setUp(self):
        self.em = EventManager()
        self.event1 = Event(Always(), lambda context, data: None)
        self.event2 = Event(Always(), lambda context, data: None)

    def test_add_event(self):
        self.em.add_event(self.event1)
        self.assertEqual(len(self.em._events), 1)

    def test_add_event_prepend(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2, prepend=True)
        self.assertEqual([self.event2, self.event1], self.em._events)

    def test_add_event_append(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2)
        self.assertEqual([self.event1, self.event2], self.em._events)

    def test_checks_should_trigger(self):
        class CountingRule(Always):
            count = 0

            def should_trigger(self, dt):
                CountingRule.count += 1
                return True

        for r in [CountingRule] * 5:
                self.em.add_event(
                    Event(r(), lambda context, data: None, check_args=False)
                )

        self.em.handle_data(None, None, datetime.datetime.now())

        self.assertEqual(CountingRule.count, 5)


class TestEventRule(TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            EventRule()

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            super(Always, Always()).should_trigger('a')


class RuleTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment.instance()
        cls.class_ = None  # Mark that this is the base class.

    def setUp(self):
        # Select a random sample of 5 trading days
        self.trading_days = self._get_random_days(5)

    def _get_random_days(self, n):
        """
        Returns a random selection n trading days.
        """
        index = random.sample(range(len(self.env.trading_days)), n)
        test_dts = (self.env.trading_days[i] for i in index)
        return (self.env.market_minutes_for_day(dt) for dt in test_dts)

    @property
    def minutes(self):
        for d in self.trading_days:
            for m in d:
                yield m.to_datetime()

    def test_completeness(self):
        """
        Tests that all rules are being tested.
        """
        if not self.class_:
            return  # This is the base class testing, it is always complete.

        dem = {
            k for k, v in vars(events_module).iteritems()
            if isinstance(v, type)
            and issubclass(v, self.class_)
            and v is not self.class_
        }
        ds = {
            k[5:] for k in dir(self)
            if k.startswith('test') and k[5:] in dem
        }
        self.assertTrue(
            dem <= ds,
            msg='This suite is missing tests for the following classes:\n' +
            '\n'.join(map(repr, dem - ds)),
        )


class TestStatelessRules(RuleTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestStatelessRules, cls).setUpClass()

        cls.class_ = StatelessRule

        cls.sept_days = cls.env.days_in_range(
            np.datetime64(datetime.date(year=2014, month=9, day=1)),
            np.datetime64(datetime.date(year=2014, month=9, day=30)),
        )

        cls.sept_week = cls.env.minutes_for_days_in_range(
            datetime.date(year=2014, month=9, day=21),
            datetime.date(year=2014, month=9, day=26),
        )

    def test_Always(self):
        should_trigger = Always().should_trigger
        self.assertTrue(all(map(should_trigger, self.minutes)))

    def test_Never(self):
        should_trigger = Never().should_trigger
        self.assertFalse(any(map(should_trigger, self.minutes)))

    def test_InvertedRule(self):
        rule = Always()
        should_trigger = rule.should_trigger
        should_not_trigger = InvertedRule(rule).should_trigger
        f = lambda m: should_trigger(m) != should_not_trigger(m)
        self.assertTrue(all(map(f, self.minutes)))

        # Test the syntax.
        self.assertIsInstance(~Always(), InvertedRule)

    def test_AfterOpen(self):
        should_trigger = AfterOpen(minutes=5, hours=1).should_trigger
        for d in self.trading_days:
            for m in islice(d, 65):
                self.assertFalse(should_trigger(m))
            for m in islice(d, 65, None):
                self.assertTrue(should_trigger(m))

    def test_BeforeClose(self):
        should_trigger = BeforeClose(hours=1, minutes=5).should_trigger
        for d in self.trading_days:
            for m in d[0:-65]:
                self.assertFalse(should_trigger(m))
            for m in d[-65:]:
                self.assertTrue(should_trigger(m))

    def test_OnDate(self):
        first_day = next(self.trading_days)
        should_trigger = OnDate(first_day[0].date()).should_trigger
        self.assertTrue(all(map(should_trigger, first_day)))
        self.assertFalse(any(map(should_trigger, self.minutes)))

    def _test_before_after_date(self, class_, op):
        minutes = list(self.minutes)
        half = int(len(minutes) / 2)
        should_trigger = class_(minutes[half].date()).should_trigger
        for m in minutes:
            if op(m.date(), minutes[half].date()):
                self.assertTrue(should_trigger(m))
            else:
                self.assertFalse(should_trigger(m))

    def test_BeforeDate(self):
        self._test_before_after_date(BeforeDate, operator.lt)

    def test_AfterDate(self):
        self._test_before_after_date(AfterDate, operator.gt)

    def test_AtTime(self):
        time = datetime.time(hour=15, minute=5)
        should_trigger = AtTime(time).should_trigger

        hit = []
        f = lambda m: should_trigger(m) == (m.time() == time) \
            and (hit.append(None) or True)
        self.assertTrue(all(map(f, self.minutes)))
        # Make sure we actually had a bar that is the time we wanted.
        self.assertTrue(hit)

    def _test_before_after_time(self, class_, op):
        time = datetime.time(hour=15, minute=5)
        should_trigger = class_(time).should_trigger

        for m in self.minutes:
            if op(m.time(), time):
                self.assertTrue(should_trigger(m))
            else:
                self.assertFalse(should_trigger(m))

    def test_BeforeTime(self):
        self._test_before_after_time(BeforeTime, operator.lt)

    def test_AfterTime(self):
        self._test_before_after_time(AfterTime, operator.gt)

    def test_HalfDay(self):
        should_trigger = HalfDay().should_trigger
        self.assertTrue(should_trigger(HALF_DAY))
        self.assertFalse(should_trigger(FULL_DAY))

    def test_NotHalfDay(self):
        should_trigger = NotHalfDay().should_trigger
        self.assertTrue(should_trigger(FULL_DAY))
        self.assertFalse(should_trigger(HALF_DAY))

    @parameterized.expand(param_range(5))
    def test_NthTradingDayOfWeek(self, n):
        should_trigger = NthTradingDayOfWeek(n).should_trigger
        prev_day = None
        n_tdays = 0
        for m in dropwhile(lambda n: not should_trigger(n), self.sept_week):
            if should_trigger(m):
                self.assertEqual(n_tdays, n)
            else:
                self.assertNotEqual(n_tdays, n)

            if not prev_day or prev_day < m.date():
                n_tdays += 1
            prev_day = m.date()

    @parameterized.expand(param_range(5))
    def test_NDaysBeforeLastTradingDayOfWeek(self, n):
        should_trigger = NDaysBeforeLastTradingDayOfWeek(n).should_trigger
        for m in self.sept_week:
            if should_trigger(m):
                n_tdays = 0
                date = m.to_datetime().date()
                next_date = self.env.next_trading_day(date)
                while next_date.day > date.day:
                    date = next_date
                    next_date = self.env.next_trading_day(date)
                    n_tdays += 1

                self.assertEqual(n_tdays, n)

    @parameterized.expand(param_range(30))
    def test_NthTradingDayOfMonth(self, n):
        should_trigger = NthTradingDayOfMonth(n).should_trigger
        for n_tdays, d in enumerate(self.sept_days):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_tdays, n)
                else:
                    self.assertNotEqual(n_tdays, n)

    @parameterized.expand(param_range(30))
    def test_NDaysBeforeLastTradingDayOfMonth(self, n):
        should_trigger = NDaysBeforeLastTradingDayOfMonth(n).should_trigger
        for n_days_before, d in enumerate(reversed(self.sept_days)):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_days_before, n)
                else:
                    self.assertNotEqual(n_days_before, n)

    @parameterized.expand([
        ('and', operator.and_, lambda t: t._test_composed_and),
        ('or', operator.or_, lambda t: t._test_composed_or),
        ('xor', operator.xor, lambda t: t._test_composed_xor),
    ])
    def test_ComposedRule(self, name, composer, tester):
        rule1 = Always()
        rule2 = Never()

        composed = composer(rule1, rule2)
        self.assertIsInstance(composed, ComposedRule)
        self.assertIs(composed.first, rule1)
        self.assertIs(composed.second, rule2)
        tester(self)(composed)

    def _test_composed_and(self, rule):
        self.assertFalse(any(map(rule.should_trigger, self.minutes)))

    def _test_composed_or(self, rule):
        self.assertTrue(all(map(rule.should_trigger, self.minutes)))

    def _test_composed_xor(self, rule):
        self.assertTrue(all(map(rule.should_trigger, self.minutes)))


class TestStatefulRules(RuleTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestStatefulRules, cls).setUpClass()

        cls.class_ = StatefulRule

    @parameterized.expand(param_range(5))
    def test_DoNTimes(self, n):
        rule = DoNTimes(n)
        min_gen = self.minutes

        for n in range(n):
            self.assertTrue(rule.should_trigger(next(min_gen)))

        self.assertFalse(any(map(rule.should_trigger, min_gen)))

    @parameterized.expand(param_range(5))
    def test_SkipNTimes(self, n):
        rule = SkipNTimes(n)
        min_gen = self.minutes

        for n in range(n):
            self.assertFalse(rule.should_trigger(next(min_gen)))

        self.assertTrue(any(map(rule.should_trigger, min_gen)))

    @parameterized.expand(
        product(range(5), [('B', 5), ('W', 10), ('M', 50), ('Q', 50)])
    )
    def test_NTimesPerPeriod(self, n, period_ndays):
        period, ndays = period_ndays
        self.trading_days = self._get_random_days(ndays)

        rule = NTimesPerPeriod(n=n, freq=period)

        minutes = list(self.minutes)
        hit = pd.Series(
            0,
            pd.date_range(minutes[0].date(), minutes[-1].date(), freq=period),
        )

        for m in self.minutes:
            if rule.should_trigger(m):
                hit[m] += 1

        for h in hit:
            self.assertLessEqual(h, n)

    def test_RuleFromCallable(self):
        rule = RuleFromCallable(lambda dt: True)
        self.assertTrue(all(map(rule.should_trigger, self.minutes)))
