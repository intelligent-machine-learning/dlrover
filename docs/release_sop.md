# DLRover Releasing SOP(For Maintainers)

1. Checkout a new branch from master named as 'rc_{version}', 
    such as: rc_0.5.0.
2. Use the aforementioned branch as the target branch for tagging, confirm the 
   release version and the release content: 
   [draft](https://github.com/intelligent-machine-learning/dlrover/releases).
3. (Use 'rc_{version}' branch) Ensure all changes have been completed, and 
   update the version number in `setup.py` to the final release version, 
   e.g., `0.5.0`.  
4. (Use 'rc_{version}' branch) Build and upload the release wheel:
    ```shell
    docker run -v `pwd`:/dlrover -it easydl/dlrover:ci bash
    
    # enter container
    cd /dlrover
    sh scripts/build_wheel.sh
    
    # upload the wheel
    twine upload -r pypi dist/dlrover-xxx-any.whl
    ```
5. Publish the release [draft](https://github.com/intelligent-machine-learning/dlrover/releases).
6. Trigger(automatically) the image release actions.