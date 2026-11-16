- **G22 (the camera's tree against brute force) no longer judges a ray the float32 crossing test cannot place.** A ray passing a
  far, grazing triangle's edge within the rounding of its barycentric coordinates (about `4 eps (distance / edge) (edge^2 / 2 area) /
  cos` of incidence: the sheared corners carry the error of the distance, not of the triangle) may lose or keep a fold's two
  crossings together; the oracle now leaves such rays out by that derived bound, where it left out only a fixed `1e-4`.
  A test change; the renderer is unchanged.
