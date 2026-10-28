# The package READMEs' lines that carried a number token when the number rule landed: its frozen start, `file
# tokens  #k  # present at D1-min`, keyed as number_allowlist.txt is (the file, the line's number tokens in order, and
# its occurrence index among that file's lines carrying the same tokens). It only shrinks. A wording edit keeps a
# line's entry; a line whose tokens change leaves this list, and passes or takes its own reason in
# number_allowlist.txt. `mime docs check` fails on a README line listed nowhere, and on a listed key that matches no
# line (delete its line); with `--base <revision>` (CI's is origin/v0.3), on a line the base revision's list does not
# hold, so a README's new number takes its own line, with its own reason, in number_allowlist.txt. Its length is
# printed, never stored.
src/mime/README.md  0.2  #1                                                                        # present at D1-min
src/mime/README.md  0.3  #1                                                                        # present at D1-min
src/mime/README.md  4  #1                                                                          # present at D1-min
src/mime/README.md  0.2 4a  #1                                                                     # present at D1-min
src/mime/README.md  2.0  #1                                                                        # present at D1-min
src/mime/README.md  4b  #1                                                                         # present at D1-min
src/mime/README.md  4c1  #1                                                                        # present at D1-min
src/mime/README.md  4c2  #1                                                                        # present at D1-min
src/mime/README.md  4c3  #1                                                                        # present at D1-min
src/mime/README.md  4c3 4c3  #1                                                                    # present at D1-min
src/mime/README.md  4c3  #2                                                                        # present at D1-min
src/mime/README.md  0.3  #2                                                                        # present at D1-min
src/mime/build/README.md  0  #1                                                                    # present at D1-min
src/mime/cli/README.md  0 1 2  #1                                                                  # present at D1-min
src/mime/cli/README.md  0 1 2  #2                                                                  # present at D1-min
src/mime/control/README.md  1 2 3 4  #1                                                            # present at D1-min
src/mime/core/README.md  0.2  #1                                                                   # present at D1-min
src/mime/core/README.md  0.2  #2                                                                   # present at D1-min
src/mime/core/README.md  0.2  #3                                                                   # present at D1-min
src/mime/core/README.md  0.2  #4                                                                   # present at D1-min
src/mime/core/multibody/README.md  256  #1                                                         # present at D1-min
src/mime/core/multibody/README.md  2008  #1                                                        # present at D1-min
src/mime/core/multibody/README.md  -1  #1                                                          # present at D1-min
src/mime/data/README.md  0.2  #1                                                                   # present at D1-min
src/mime/data/README.md  151,887  #1                                                               # present at D1-min
src/mime/data/README.md  1.0  #1                                                                   # present at D1-min
src/mime/data/records/README.md  256  #1                                                           # present at D1-min
src/mime/data/records/README.md  88851e62 293.15 310.15 95 2018 08 7d6a8f93  #1                    # present at D1-min
src/mime/data/records/README.md  1.32 1.37 40ff82e8  #1                                            # present at D1-min
src/mime/data/records/README.md  1e4593c23af14d28c6f6e204c192ed4ce694d7be 7cac160d 2026 10 20  #1  # present at D1-min
src/mime/data/records/README.md  2026 10 19  #1                                                    # present at D1-min
src/mime/data/records/README.md  1.0  #1                                                           # present at D1-min
src/mime/data/records/README.md  1.0  #2                                                           # present at D1-min
src/mime/data/records/README.md  2.0  #1                                                           # present at D1-min
src/mime/data/records/README.md  256  #2                                                           # present at D1-min
src/mime/docs/README.md  0.2  #1                                                                   # present at D1-min
src/mime/docs/README.md  0.3  #1                                                                   # present at D1-min
src/mime/docs/README.md  0.2 0.2  #1                                                               # present at D1-min
src/mime/dynamics/README.md  0.3  #1                                                               # present at D1-min
src/mime/dynamics/README.md  4b  #1                                                                # present at D1-min
src/mime/fields/README.md  0.2  #1                                                                 # present at D1-min
src/mime/fields/README.md  1e-4  #1                                                                # present at D1-min
src/mime/interactions/README.md  0.2 1  #1                                                         # present at D1-min
src/mime/interactions/README.md  0.655  #1                                                         # present at D1-min
src/mime/interactions/README.md  2.65 6.24  #1                                                     # present at D1-min
src/mime/interactions/README.md  0.2  #1                                                           # present at D1-min
src/mime/interactions/README.md  2  #1                                                             # present at D1-min
src/mime/interactions/README.md  16%  #1                                                           # present at D1-min
src/mime/interactions/README.md  0.2 0.32  #1                                                      # present at D1-min
src/mime/materials/README.md  256  #1                                                              # present at D1-min
src/mime/nodes/README.md  0.3  #1                                                                  # present at D1-min
src/mime/nodes/README.md  0.2  #1                                                                  # present at D1-min
src/mime/nodes/README.md  0.2  #2                                                                  # present at D1-min
src/mime/nodes/README.md  3  #1                                                                    # present at D1-min
src/mime/nodes/README.md  3  #2                                                                    # present at D1-min
src/mime/nodes/README.md  0.3  #2                                                                  # present at D1-min
src/mime/nodes/README.md  0.2  #3                                                                  # present at D1-min
src/mime/nodes/README.md  0.2  #4                                                                  # present at D1-min
src/mime/nodes/README.md  0.2  #5                                                                  # present at D1-min
src/mime/nodes/README.md  1e-12  #1                                                                # present at D1-min
src/mime/nodes/README.md  2.8e-15  #1                                                              # present at D1-min
src/mime/saved/README.md  256  #1                                                                  # present at D1-min
src/mime/saved/README.md  256  #2                                                                  # present at D1-min
src/mime/scene/README.md  256  #1                                                                  # present at D1-min
src/mime/scene/README.md  64  #1                                                                   # present at D1-min
src/mime/scene/README.md  26  #1                                                                   # present at D1-min
src/mime/surrogates/README.md  0.2  #1                                                             # present at D1-min
src/mime/surrogates/README.md  0.3  #1                                                             # present at D1-min
src/mime/surrogates/README.md  21  #1                                                              # present at D1-min
src/mime/surrogates/README.md  1e-5  #1                                                            # present at D1-min
src/mime/surrogates/README.md  0.2  #2                                                             # present at D1-min
src/mime/surrogates/README.md  21  #2                                                              # present at D1-min
src/mime/surrogates/README.md  0.33 0.25  #1                                                       # present at D1-min
src/mime/surrogates/README.md  3.5 0.2 2.33 3.7 0.25 0.655 2.65 6.24  #1                           # present at D1-min
src/mime/surrogates/README.md  0.3  #2                                                             # present at D1-min
src/mime/surrogates/README.md  0 0.12 0.246 0.655  #1                                              # present at D1-min
src/mime/surrogates/README.md  0.2 0.66 0.35  #1                                                   # present at D1-min
src/mime/surrogates/README.md  0 0.024 0.1 0.2 0.3 4.13 9.73 0.2  #1                               # present at D1-min
src/mime/surrogates/README.md  0.2 1% 16.2%  #1                                                    # present at D1-min
src/mime/surrogates/README.md  0.2 0.32  #1                                                        # present at D1-min
src/mime/testing/README.md  200  #1                                                                # present at D1-min
src/mime/testing/README.md  1e-12  #1                                                              # present at D1-min
src/mime/testing/README.md  64  #1                                                                 # present at D1-min
src/mime/testing/README.md  4e-6 15  #1                                                            # present at D1-min
src/mime/testing/README.md  4c1  #1                                                                # present at D1-min
src/mime/testing/README.md  64  #2                                                                 # present at D1-min
src/mime/testing/README.md  1e-10 1e-3  #1                                                         # present at D1-min
src/mime/testing/README.md  70  #1                                                                 # present at D1-min
src/mime/testing/README.md  210 210 50 58 60 4b  #1                                                # present at D1-min
src/mime/testing/README.md  9  #1                                                                  # present at D1-min
src/mime/testing/README.md  0  #1                                                                  # present at D1-min
src/mime/testing/README.md  0.2 1.0  #1                                                            # present at D1-min
src/mime/vocabulary/README.md  7  #1                                                               # present at D1-min
src/mime/vocabulary/README.md  6 6  #1                                                             # present at D1-min
src/mime/vocabulary/README.md  6 6 3  #1                                                           # present at D1-min
src/mime/vocabulary/README.md  3 3  #1                                                             # present at D1-min
