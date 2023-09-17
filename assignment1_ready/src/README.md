## Assignment 1

The `.png` files are under the folder `results`.

To run the code, you can simply run `run.sh` whose default is to run the image processing pipeline and show result for every step. You can also try out manual white balancing by uncommenting the corresponding lines.

Another option is to run `main.py`. The flag definitions are as follows:

- `-f`: specify the file path
- `-m`: specify the running mode
  - `processing`: run the image processing pipeline
  - `manual_balancing`: run manual white balancing
  - `find_bayer_pattern`: compare different Bayer patterns and output the result
- `-wb`: if set to `True`, show the result of white balancing
- `-wm`: specify the white balancing algorithm.
  - `max`: white world white balancing algorithm
  - `mean`: gray world white balancing algorithm
  - `scale`: use `<r_scale>`, `<g_scale>` and `<b_scale>`
- `-d`: if set to `True`, show the result of demosaicing
- `-c`: if set to `True`, show the result of color correction
- `-b`: if set to `True`, show the result of brightness adjustment
- `-pi`: specify the post-brightening mean gray intensity. Default is 0.3
- `-g`: if set to `True`, show the result of gamma encoding

