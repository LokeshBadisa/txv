.. txv documentation master file, created by
   sphinx-quickstart on Thu Aug  1 21:18:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

txv documentation
=================

**Version:** 0.0.1


**Useful Links:** `Installation <https://pypi.org/project/txv/>`_ | `Source Repository <https://github.com/LokeshBadisa/txv>`_ | `Issue Tracker <https://github.com/LokeshBadisa/txv/issues>`_ 


txv is a vision transformers explainability package. 
It provides CAM like visualization for vision transformers. 
It is built on the top of `transformerinterp <https://github.com/jiaminchen-1031/transformerinterp>`_ 
and `Transformer-Explainability <https://github.com/hila-chefer/Transformer-Explainability>`_ repositories. 


Note that txv works for models defined in :doc:`vit` only.

.. toctree::
    :maxdepth: 2
    :hidden:

    api_reference
    tutorials
    

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :text-align: center

        API reference
        ^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in txv. The reference describes how the
        methods work and which parameters can be used.

        +++

        .. button-ref:: api_reference
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :text-align: center

        Tutorials
        ^^^

        Tutorials provide an understanding of how functions of txv should be used in unison.
        The tutorials are designed to help you get started with txv.

        +++

        .. button-ref:: tutorials
            :expand:
            :color: secondary
            :click-parent:

            To the tutorials





