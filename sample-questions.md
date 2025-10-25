## Ingredients

1. **"Find foods that contain milk"**
   - **Uses**: `contains_milk` column
   - **Returns**: Foods flagged as containing milk

2. **"What foods contain eggs?"**
   - **Uses**: `contains_eggs` column
   - **Returns**: Foods flagged as containing eggs

3. **"Show me foods that contain peanuts"**
   - **Uses**: `contains_peanuts` column
   - **Returns**: Foods flagged as containing peanuts

4. **"Find foods with tree nuts"**
   - **Uses**: `contains_tree_nuts` column
   - **Returns**: Foods flagged as containing tree nuts

5. **"What foods contain fish?"**
   - **Uses**: `contains_fish` column
   - **Returns**: Foods flagged as containing fish

6. **"Show me foods that contain shellfish"**
   - **Uses**: `contains_shellfish` column
   - **Returns**: Foods flagged as containing shellfish

7. **"Find foods that contain soy"**
   - **Uses**: `contains_soy` column
   - **Returns**: Foods flagged as containing soy

8. **"What foods contain wheat or gluten?"**
   - **Uses**: `contains_wheat` and `contains_gluten` columns
   - **Returns**: Foods flagged as containing wheat or gluten

9. **"Show me foods that contain sesame"**
   - **Uses**: `contains_sesame` column
   - **Returns**: Foods flagged as containing sesame

10. **"Find foods with sulfites, celery, or mustard"**
    - **Uses**: `contains_sulfites`, `contains_celery`, and `contains_mustard` columns
    - **Returns**: Foods flagged as containing any of these allergens

## Categories

### Meal Type Queries

1. **"Find foods suitable for breakfast"**
   - **Uses**: `is_breakfast` column
   - **Returns**: Foods flagged as breakfast options

2. **"What are some good snacks?"**
   - **Uses**: `is_snack` column
   - **Returns**: Foods flagged as snacks

3. **"Show me foods suitable for dinner but not desserts"**
   - **Uses**: `is_dinner` and `is_dessert` columns
   - **Returns**: Dinner options excluding desserts

### Dietary Compatibility Queries

4. **"Find vegan-compatible foods"**
   - **Uses**: `is_vegan_compatible` column
   - **Returns**: Foods flagged as vegan-compatible

5. **"What foods are keto-compatible and gluten-free?"**
   - **Uses**: `is_keto_compatible` and `is_gluten_free_compatible` columns
   - **Returns**: Foods that meet both keto and gluten-free criteria

6. **"Show me Mediterranean-compatible foods that are also dairy-free"**
   - **Uses**: `is_mediterranean_compatible` and `is_dairy_free_compatible` columns
   - **Returns**: Foods suitable for a Mediterranean diet and free of dairy

### Food Category Queries

7. **"What are some dairy products?"**
   - **Uses**: `is_dairy` column
   - **Returns**: Foods flagged as dairy products

8. **"Find meat or fish options"**
   - **Uses**: `is_meat_fish` column
   - **Returns**: Foods flagged as meat or fish

9. **"What are some grains or cereals?"**
   - **Uses**: `is_grains_cereals` column
   - **Returns**: Foods flagged as grains or cereals

10. **"Show me fruits and vegetables"**
    - **Uses**: `is_fruits_vegetables` column
    - **Returns**: Foods flagged as fruits or vegetables

### Prepared and Processed Foods Queries

11. **"Find prepared meals"**
    - **Uses**: `is_prepared_meals` column
    - **Returns**: Foods flagged as prepared meals

12. **"What are some condiments or spices?"**
    - **Uses**: `is_condiments_spices` column
    - **Returns**: Foods flagged as condiments or spices