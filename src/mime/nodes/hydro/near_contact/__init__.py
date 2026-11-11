"""The near-contact law, one for every rung that hands a near contact over: a rung declares the gap
it resolves down to and evaluates itself with the contact opened to it (:func:`opening`); the law supplies the
increment from there to the true gap from the contact's local geometry (:class:`NearContact`) and the body's motion
(:func:`increment`), by the form the plan picks from the body's kind (``Geometry.near_contact``): a sphere's exact
increment (``"sphere"``: the plane's exact remainder from MIME's table of the bispherical solutions, with the contact's
own singular terms, so exact against a plane and the hybrid in a tube), or the thin film's (``"film"``, every other
point contact, MIME-ANO-154 open for it). The roughness cutoff below which a contact carries the gap
(:func:`roughness_cutoff`) and the ambient flow's zero on the opened wall (:func:`ambient_point`) are the seam's, the
same for every rung. ``_seam``'s docstring holds the contract, ``_sphere``'s the sphere's form and its sources."""

from mime.nodes.hydro.near_contact._seam import (
    AMBIENT_ZERO,
    CURVATURE_FLOOR,
    FORMS,
    LIFT_WIDTH,
    ROUGHNESS_WIDTH,
    Increment,
    NearContact,
    ambient_point,
    film_increment,
    floored,
    increment,
    opening,
    roughness_cutoff,
)

__all__ = ["AMBIENT_ZERO", "CURVATURE_FLOOR", "FORMS", "LIFT_WIDTH", "ROUGHNESS_WIDTH", "Increment", "NearContact",
           "ambient_point", "film_increment", "floored", "increment", "opening", "roughness_cutoff"]
