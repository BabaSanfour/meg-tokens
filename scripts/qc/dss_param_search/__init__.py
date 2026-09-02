"""Parameter search for the three DSS stages of the ``dss`` cleaning stream.

The stages are searched one at a time with the upstream stages fixed at their
production values, because that is the input each stage actually sees in
``clean_dss``. Scoring each stage on the bare trunk instead would make the
numbers cheaper to produce and incomparable to production -- and to the arena's
own measurements, which scored the muscle stage after cardiac and ocular had
already been removed.
"""
