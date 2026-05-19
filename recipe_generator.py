#!/usr/bin/env python3
"""
Recipe Generator Integration - Runs on Raspberry Pi 5.
Receives fruit classification and generates recipes via API.
Integrates with SpoonacularAPI or local recipe database.
"""

import requests
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class RecipeGenerator:
    """Generate recipes for detected fruits."""
    
    def __init__(self, api_key: Optional[str] = None, use_local: bool = True):
        """
        Initialize recipe generator.
        
        Args:
            api_key: Spoonacular API key (optional)
            use_local: Use local JSON recipe database if True
        """
        self.api_key = api_key or os.getenv('SPOONACULAR_API_KEY')
        self.use_local = use_local
        self.base_url = "https://api.spoonacular.com"
        
        if self.use_local:
            self.recipes_db = self._load_local_recipes()
        
    def _load_local_recipes(self) -> Dict:
        """Load local recipe database."""
        recipes_file = 'recipes_db.json'
        
        if os.path.exists(recipes_file):
            with open(recipes_file, 'r') as f:
                return json.load(f)
        
        # Default recipe database
        return {
            'apple': [
                {'name': 'Apple Pie', 'ingredients': ['apples', 'sugar', 'flour', 'butter'], 'prep_time': 60},
                {'name': 'Apple Salad', 'ingredients': ['apples', 'lettuce', 'vinegar'], 'prep_time': 15},
                {'name': 'Applesauce', 'ingredients': ['apples', 'sugar', 'cinnamon'], 'prep_time': 30},
            ],
            'banana': [
                {'name': 'Banana Bread', 'ingredients': ['bananas', 'flour', 'sugar', 'eggs'], 'prep_time': 50},
                {'name': 'Banana Smoothie', 'ingredients': ['bananas', 'milk', 'honey'], 'prep_time': 5},
                {'name': 'Banana Pancakes', 'ingredients': ['bananas', 'eggs', 'flour'], 'prep_time': 20},
            ],
            'orange': [
                {'name': 'Orange Juice', 'ingredients': ['oranges'], 'prep_time': 10},
                {'name': 'Orange Chicken', 'ingredients': ['oranges', 'chicken', 'soy_sauce'], 'prep_time': 40},
                {'name': 'Orange Cake', 'ingredients': ['oranges', 'flour', 'sugar', 'eggs'], 'prep_time': 60},
            ],
            'strawberry': [
                {'name': 'Strawberry Jam', 'ingredients': ['strawberries', 'sugar', 'lemon'], 'prep_time': 45},
                {'name': 'Strawberry Shortcake', 'ingredients': ['strawberries', 'cake', 'whipped_cream'], 'prep_time': 30},
                {'name': 'Strawberry Smoothie', 'ingredients': ['strawberries', 'milk', 'yogurt'], 'prep_time': 5},
            ],
        }
    
    def get_recipes_local(self, fruit: str, max_recipes: int = 3) -> List[Dict]:
        """Get recipes from local database."""
        fruit_lower = fruit.lower()
        
        if fruit_lower in self.recipes_db:
            return self.recipes_db[fruit_lower][:max_recipes]
        
        return []
    
    def get_recipes_api(self, fruit: str, max_recipes: int = 3) -> List[Dict]:
        """Get recipes from Spoonacular API."""
        if not self.api_key:
            print("⚠ No API key provided. Using local recipes.")
            return self.get_recipes_local(fruit, max_recipes)
        
        try:
            url = f"{self.base_url}/recipes/findByIngredients"
            params = {
                'ingredients': fruit,
                'number': max_recipes,
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            recipes = response.json()
            return [
                {
                    'name': r.get('title', 'Unknown'),
                    'id': r.get('id'),
                    'image': r.get('image'),
                    'used_ingredients': r.get('usedIngredients', []),
                }
                for r in recipes
            ]
        
        except requests.exceptions.RequestException as e:
            print(f"⚠ API error: {e}. Using local recipes.")
            return self.get_recipes_local(fruit, max_recipes)
    
    def get_recipes(self, fruit: str, max_recipes: int = 3) -> List[Dict]:
        """Get recipes for a fruit."""
        if self.use_local:
            return self.get_recipes_local(fruit, max_recipes)
        else:
            return self.get_recipes_api(fruit, max_recipes)
    
    def format_recipes(self, fruit: str, recipes: List[Dict]) -> str:
        """Format recipes for display."""
        output = f"\n{'='*70}\n"
        output += f"🍎 RECIPES FOR {fruit.upper()}\n"
        output += f"{'='*70}\n\n"
        
        if not recipes:
            output += "No recipes found.\n"
            return output
        
        for i, recipe in enumerate(recipes, 1):
            output += f"{i}. {recipe.get('name', 'Unknown Recipe')}\n"
            
            if 'prep_time' in recipe:
                output += f"   ⏱ Prep time: {recipe['prep_time']} min\n"
            
            if 'ingredients' in recipe:
                output += f"   📝 Ingredients: {', '.join(recipe['ingredients'][:3])}\n"
            
            output += "\n"
        
        output += f"{'='*70}\n\n"
        return output

class IngredientSearch:
    """Search and manage ingredient inventory."""
    
    def __init__(self):
        self.inventory = {}
        self.load_inventory()
    
    def load_inventory(self):
        """Load ingredient inventory from file."""
        inventory_file = 'inventory.json'
        if os.path.exists(inventory_file):
            with open(inventory_file, 'r') as f:
                self.inventory = json.load(f)
    
    def save_inventory(self):
        """Save inventory to file."""
        with open('inventory.json', 'w') as f:
            json.dump(self.inventory, f, indent=2)
    
    def add_ingredient(self, fruit: str, quantity: int = 1):
        """Add detected fruit to inventory."""
        fruit_lower = fruit.lower()
        self.inventory[fruit_lower] = self.inventory.get(fruit_lower, 0) + quantity
        self.save_inventory()
        print(f"✓ Added {fruit} to inventory (total: {self.inventory[fruit_lower]})")
    
    def get_recipes_for_inventory(self, recipe_generator: RecipeGenerator) -> Dict:
        """Get recipes based on available ingredients."""
        all_recipes = {}
        
        for fruit in self.inventory:
            recipes = recipe_generator.get_recipes(fruit, max_recipes=2)
            if recipes:
                all_recipes[fruit] = recipes
        
        return all_recipes
    
    def suggest_meals(self) -> str:
        """Suggest meals based on inventory."""
        output = "\n" + "="*70 + "\n"
        output += "📦 CURRENT INVENTORY\n"
        output += "="*70 + "\n\n"
        
        if not self.inventory:
            output += "No ingredients in inventory.\n"
        else:
            for fruit, quantity in sorted(self.inventory.items()):
                output += f"  • {fruit.capitalize()}: {quantity}\n"
        
        output += "\n" + "="*70 + "\n\n"
        return output

def demo():
    """Demo recipe generation."""
    print("\n" + "="*70)
    print("RECIPE GENERATOR DEMO")
    print("="*70 + "\n")
    
    generator = RecipeGenerator(use_local=True)
    inventory = IngredientSearch()
    
    # Simulate detections
    detected_fruits = ['apple', 'banana', 'strawberry']
    
    for fruit in detected_fruits:
        print(f"Processing: {fruit}")
        inventory.add_ingredient(fruit)
        
        recipes = generator.get_recipes(fruit, max_recipes=3)
        formatted = generator.format_recipes(fruit, recipes)
        print(formatted)

if __name__ == '__main__':
    demo()
