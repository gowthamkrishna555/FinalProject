var karnataka = ee.Geometry.Polygon([
  [74.0503, 13.1587],
  [77.3535, 13.1587],
  [77.3535, 16.7875],
  [74.0503, 16.7875],
  [74.0503, 13.1587]
]);

// Load Watershed Basins data for Karnataka
var watershed = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level12")
  .filterBounds(karnataka);

// Extract latitude and longitude information
var karnatakaWithLatLng = watershed.map(function(feature) {
  var centroid = feature.geometry().centroid();
  var latitude = centroid.coordinates().get(1);
  var longitude = centroid.coordinates().get(0);

  // Simulate labeling based on your wetland presence criteria
  var wetlandPresent = ee.Number.parse(latitude).gt(14.5).and(ee.Number.parse(longitude).lt(75));

  // Add a new property 'wetland_label' to the feature
  return feature.set('latitude', latitude)
                .set('longitude', longitude)
                .set('wetland_label', wetlandPresent ? 1 : 0);
});

// Print the first feature to check the results
print('First Feature:', karnatakaWithLatLng.first());

// Export the Karnataka boundary with latitude, longitude, and wetland label to CSV
Export.table.toDrive({
  collection: karnatakaWithLatLng,
  description: 'Karnataka_Watershed_Labeled',
  fileFormat: 'CSV'
});