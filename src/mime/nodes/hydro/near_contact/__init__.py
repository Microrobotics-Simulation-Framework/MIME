"""The near-contact law, one for every rung that hands a near contact over: a rung declares the gap
it resolves down to and evaluates itself with its contacts' opposing surface opened to it (:func:`surface_opening`);
the law supplies the increment from there to each true gap from the contact's local geometry (:class:`NearContact`,
several held as :class:`NearContacts`, found by the seam's own search, ``_search``) and the body's motion
(:func:`increment`, summed by :func:`summed`), by the form the plan picks from the body's kind
(``Geometry.near_contact``): a sphere's exact increment (``"sphere"``: the plane's exact remainder from MIME's table of
the bispherical solutions, with the contact's own singular terms, so exact against a plane and the hybrid in a tube),
or the elliptic point contact's (``"elliptic"``, every other kind: its singular terms from its curvatures alone,
MIME-ANO-154 open for it). Each contact is classed (:func:`classified`): the forms serve point contacts; a line or a
ring has no form yet. The roughness cutoff below which a contact carries the gap (:func:`roughness_cutoff`) and the
ambient flow's zero on the opened wall (:func:`ambient_point`) are the seam's, the same for every rung. ``_seam``'s
docstring holds the contract, ``_sphere``'s and ``_elliptic``'s the forms and their sources."""

from mime.nodes.hydro.near_contact._seam import (
    AMBIENT_ZERO,
    CLASSES,
    CURVATURE_FLOOR,
    FORMS,
    LIFT_WIDTH,
    MERGE_SHARE,
    RING_SHARE,
    ROUGHNESS_WIDTH,
    SOFT_SHARE,
    Increment,
    NearContact,
    NearContacts,
    ambient_point,
    classified,
    floored,
    in_band,
    increment,
    merged,
    opening,
    roughness_cutoff,
    shares,
    summed,
    surface_opening,
)

__all__ = ["AMBIENT_ZERO", "CLASSES", "CURVATURE_FLOOR", "FORMS", "LIFT_WIDTH", "MERGE_SHARE", "RING_SHARE",
           "ROUGHNESS_WIDTH", "SOFT_SHARE", "Increment", "NearContact", "NearContacts", "ambient_point", "classified",
           "floored", "in_band", "increment", "merged", "opening", "roughness_cutoff", "shares", "summed",
           "surface_opening"]
