# Changelog

### 0.1.0-beta (Last Commit: 8/6/20)

- `Vector`
    - General-use runtime has been decreased.
- `VectorField`
    - General-use runtime has been decreased.
    - Removed `cmap_func` argument from `plot()`. 
        - The mapping of colors will always be based on the vector magnitudes. 
        - Author's note: this change was made for sake of optimization, and since curl and divergence is typically represented by a heatmap in most practical applications (possible upcoming feature).
    - Added `blit` argument to `particles()`.
        - Type: `bool`. Option to blit the animation. 
        - Should be set to `False` if `plot()` is active with interactivity enabled.
        - Author's note: whether the animation was blitted or not was previously dependent on if `plot()` had interactivity enabled. Some backend optimizations had to be made that prevented `particles()` from knowing `plot()` states, and so `blit` is being left for the user to control.
    - The `scale` parameter of `plot()` no longer influences the speed of particles in the `particles()` method animation. 
        - Author's note: this is a consequence of the optimization explained above. Matplotlib isn't cut-out for dealing with this kind of animation work anyways.
- Backend Changes
    - Private methods having to do with interactivity previously contained within `Vector` and `VectorField` have been moved into their own file, handlers.py.
        - There are four new classes: `_MPLPlate`, `_VectorEventHandler`, `_VectorFieldEventHandler`, and `_ParticleSimulationHandler`. 
        - All handler classes inherit from `_MPLPlate`.
        - Classes are private.
        - All private attributes previously in `Vector` and `VectorField` that had to do with plotting have been moved into the handler classes.
            - Consequently improves runtime by a huge amount!
        - Most algorithms remain the same, though with a few simplifications here and there due to less dependencies crossing between `VectorField`'s `particles()` and `plot()` methods.
    - Collaborators added to file docstrings (sorry about that!).
    - Re-wording of many method docstrings.


### 0.0.1-beta (Last Commit: 7/1/20)

Initial version.
