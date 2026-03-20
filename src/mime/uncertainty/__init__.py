"""MIME uncertainty layer — sensing and actuation uncertainty models.

The UncertaintyModel sits at the boundary between true simulation state
and the controller. It has two channels:
- observe(): sensing uncertainty (what the controller sees)
- actuate(): actuation uncertainty (what the physics receives)

Models compose via + operator: (model_a + model_b) applies both in sequence.
"""
