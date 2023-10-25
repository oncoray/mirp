Configure the image processing and feature extraction workflow
==============================================================

MIRP implements the standardised image processing and feature extraction workflow recommended by the Image Biomarker
Standardization Initiative. Many aspects of this workflow can be configured. This can be done in several ways:

* Using keyword arguments. The keyword arguments match the parameters used to initialise the various settings objects
  documented below.
* By creating a :class:`~mirp.settings.settingsGeneric.SettingsClass` object. This object can be initialised using the
  same keyword arguments as above. Alternatively, the attributes of the
  :class:`~mirp.settings.settingsGeneric.SettingsClass` can be filled with the specific objects documented below.
* By specifying the configuration in a stand-alone settings ``xml`` file. An empty copy of the ``xml`` file can be
  created using :func:`mirp.utilities.config_utilities.get_settings_xml`.

General settings
----------------

.. automodule:: mirp.settings.settingsGeneral
   :members:
   :no-undoc-members:
   :show-inheritance:

Image processing settings
-------------------------

.. automodule:: mirp.settings.settingsImageProcessing
   :members:
   :no-undoc-members:
   :show-inheritance:

Image perturbation settings
---------------------------

.. automodule:: mirp.settings.settingsPerturbation
   :members:
   :no-undoc-members:
   :show-inheritance:

Image interpolation settings
----------------------------

.. automodule:: mirp.settings.settingsInterpolation
   :members:
   :no-undoc-members:
   :show-inheritance:

Mask resegmentation settings
----------------------------

.. automodule:: mirp.settings.settingsMaskResegmentation
   :members:
   :no-undoc-members:
   :show-inheritance:

Feature computation settings
----------------------------

.. automodule:: mirp.settings.settingsFeatureExtraction
   :members:
   :no-undoc-members:
   :show-inheritance:

Image transformation settings
-----------------------------

.. automodule:: mirp.settings.settingsImageTransformation
   :members:
   :no-undoc-members:
   :show-inheritance:

Generic settings object
-----------------------

.. automodule:: mirp.settings.settingsGeneric
   :members:
   :no-undoc-members:
   :show-inheritance:

Creating a settings xml file
----------------------------

.. autofunction:: mirp.utilities.config_utilities.get_settings_xml
