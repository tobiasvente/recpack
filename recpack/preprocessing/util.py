# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert



def rescale_id_space(ids, id_mapping=None):
    """Map identifiers to consecutive integer indices.

    Existing entries in ``id_mapping`` are preserved. Previously unseen values
    are inserted in iteration order, starting one above the largest existing
    index (or at zero for a new mapping). Duplicate identifiers are ignored.

    :param ids: Identifiers to add to the mapping.
    :type ids: Iterable[Hashable]
    :param id_mapping: Existing identifier-to-index mapping to extend. When
        provided, the dictionary is modified in place.
    :type id_mapping: dict, optional
    :return: The extended mapping, or a newly created mapping if none was
        supplied.
    :rtype: dict
    """
    counter = 0

    if id_mapping is not None and len(id_mapping) > 0:
        counter = max(id_mapping.values()) + 1
    else:
        id_mapping = {}
    for val in ids:
        if val not in id_mapping:
            id_mapping[val] = counter
            counter += 1

    return id_mapping
