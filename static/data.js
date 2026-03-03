// ===== CATEGORIES =====
const CATEGORIES = [
  { id: 1, name: 'Pizza', emoji: '🍕', color: '#FF6B35' },
  { id: 2, name: 'Biryani', emoji: '🍛', color: '#E63946' },
  { id: 3, name: 'Burger', emoji: '🍔', color: '#F4A261' },
  { id: 4, name: 'Sushi', emoji: '🍣', color: '#2A9D8F' },
  { id: 5, name: 'Pasta', emoji: '🍝', color: '#8338EC' },
  { id: 6, name: 'Desserts', emoji: '🧁', color: '#FF85A1' },
  { id: 7, name: 'Chinese', emoji: '🥟', color: '#FB8500' },
  { id: 8, name: 'Salads', emoji: '🥗', color: '#52B788' },
  { id: 9, name: 'Sandwich', emoji: '🥪', color: '#6A994E' },
  { id: 10, name: 'Coffee', emoji: '☕', color: '#8B5E3C' },
  { id: 11, name: 'Ice Cream', emoji: '🍦', color: '#FFB347' },
  { id: 12, name: 'Tacos', emoji: '🌮', color: '#E76F51' },
];

// ===== CUISINES =====
const CUISINES = [
  { name: 'North Indian', emoji: '🍲', count: '1.2k restaurants', color: '#FF6B35' },
  { name: 'South Indian', emoji: '🥘', count: '890 restaurants', color: '#E63946' },
  { name: 'Chinese', emoji: '🥟', count: '765 restaurants', color: '#FB8500' },
  { name: 'Continental', emoji: '🥩', count: '543 restaurants', color: '#2A9D8F' },
  { name: 'Italian', emoji: '🍕', count: '432 restaurants', color: '#8338EC' },
  { name: 'Japanese', emoji: '🍣', count: '287 restaurants', color: '#FF85A1' },
  { name: 'Mexican', emoji: '🌮', count: '198 restaurants', color: '#E76F51' },
  { name: 'Thai', emoji: '🍜', count: '176 restaurants', color: '#52B788' },
];

// ===== RESTAURANTS =====
const RESTAURANTS = [
  {
    id: 1,
    name: "Spice Garden",
    cuisine: "North Indian • Mughlai • Biryani",
    rating: 4.5,
    reviews: 2845,
    time: "30-40 min",
    price: "₹300 for two",
    priceNum: 300,
    distance: "1.2 km",
    image: "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400&q=80",
    tags: ["Best Seller", "Top Rated"],
    offer: "40% OFF upto ₹80",
    isVeg: false,
    featured: true,
    category: "Biryani",
    menu: [
      {
        section: "Bestsellers",
        items: [
          { id: 101, name: "Chicken Biryani", desc: "Aromatic basmati rice with tender chicken, saffron & whole spices", price: 320, rating: 4.6, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=300&q=80" },
          { id: 102, name: "Butter Chicken", desc: "Creamy tomato-based gravy with soft chicken chunks", price: 280, rating: 4.5, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=300&q=80" },
          { id: 103, name: "Dal Makhani", desc: "Slow-cooked black lentils in rich buttery gravy", price: 200, rating: 4.4, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=300&q=80" },
        ]
      },
      {
        section: "Starters",
        items: [
          { id: 104, name: "Seekh Kebab", desc: "Minced meat skewers with herbs, grilled in tandoor", price: 240, rating: 4.3, veg: false, image: "https://images.unsplash.com/photo-1599487488170-d11ec9c172f0?w=300&q=80" },
          { id: 105, name: "Paneer Tikka", desc: "Marinated cottage cheese cubes grilled in tandoor", price: 220, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=300&q=80" },
          { id: 106, name: "Veg Samosa (2 pcs)", desc: "Crispy pastry filled with spiced potatoes & peas", price: 80, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=300&q=80" },
        ]
      },
      {
        section: "Main Course",
        items: [
          { id: 107, name: "Rogan Josh", desc: "Kashmiri lamb curry with aromatic whole spices", price: 340, rating: 4.5, veg: false, image: "https://images.unsplash.com/photo-1574894709920-11b28e7367e3?w=300&q=80" },
          { id: 108, name: "Palak Paneer", desc: "Spinach gravy with fresh cottage cheese cubes", price: 220, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=300&q=80" },
          { id: 109, name: "Chicken Tikka Masala", desc: "Grilled chicken in spiced tomato-cream sauce", price: 300, rating: 4.6, veg: false, image: "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=300&q=80" },
        ]
      },
      {
        section: "Breads",
        items: [
          { id: 110, name: "Garlic Naan", desc: "Leavened bread with garlic butter, baked in tandoor", price: 60, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=300&q=80" },
          { id: 111, name: "Butter Roti (3 pcs)", desc: "Whole wheat bread with butter", price: 45, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=300&q=80" },
        ]
      },
      {
        section: "Desserts",
        items: [
          { id: 112, name: "Gulab Jamun", desc: "Soft milk solids balls soaked in rose syrup", price: 100, rating: 4.5, veg: true, image: "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=300&q=80" },
          { id: 113, name: "Phirni", desc: "Creamy rice pudding with saffron and cardamom", price: 120, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=300&q=80" },
        ]
      }
    ]
  },
  {
    id: 2,
    name: "Pizza Palace",
    cuisine: "Pizza • Italian • Pastas",
    rating: 4.3,
    reviews: 1987,
    time: "25-35 min",
    price: "₹400 for two",
    priceNum: 400,
    distance: "0.8 km",
    image: "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&q=80",
    tags: ["Trending"],
    offer: "FLAT 20% OFF",
    isVeg: false,
    featured: false,
    category: "Pizza",
    menu: [
      {
        section: "Pizzas",
        items: [
          { id: 201, name: "Margherita", desc: "Classic tomato sauce, fresh mozzarella, basil", price: 249, rating: 4.4, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=300&q=80" },
          { id: 202, name: "BBQ Chicken", desc: "Smoky BBQ sauce, grilled chicken, onions, peppers", price: 349, rating: 4.5, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=300&q=80" },
          { id: 203, name: "Veggie Supreme", desc: "Bell peppers, olives, mushrooms, onions, tomatoes", price: 299, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1565299507177-b0ac66763828?w=300&q=80" },
          { id: 204, name: "Pepperoni", desc: "Classic pepperoni with mozzarella cheese", price: 379, rating: 4.6, veg: false, image: "https://images.unsplash.com/photo-1628840042765-356cda07504e?w=300&q=80" },
        ]
      },
      {
        section: "Pastas",
        items: [
          { id: 205, name: "Penne Arrabbiata", desc: "Penne in spicy tomato garlic sauce", price: 199, rating: 4.1, veg: true, image: "https://images.unsplash.com/photo-1563379926898-05f4575a45d8?w=300&q=80" },
          { id: 206, name: "Spaghetti Carbonara", desc: "Spaghetti with egg, cheese, pancetta, black pepper", price: 279, rating: 4.4, veg: false, image: "https://images.unsplash.com/photo-1612874742237-6526221588e3?w=300&q=80" },
        ]
      },
      {
        section: "Sides & Drinks",
        items: [
          { id: 207, name: "Garlic Bread (6 pcs)", desc: "Toasted bread with garlic butter and herbs", price: 99, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1619535860434-cf9b902b1b5c?w=300&q=80" },
          { id: 208, name: "Tiramisu", desc: "Classic Italian dessert with mascarpone", price: 159, rating: 4.5, veg: true, image: "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=300&q=80" },
          { id: 209, name: "Coke (500ml)", desc: "Chilled Coca-Cola soda", price: 49, rating: 4.2, veg: true, image: "https://www.vhv.rs/viewpic/hxJiJoR_coke-bottle-500-ml-coca-cola-bottle-500/#" },
        ]
      }
    ]
  },
  {
    id: 3,
    name: "Burger Beast",
    cuisine: "Burgers • American • Shakes",
    rating: 4.2,
    reviews: 3201,
    time: "20-30 min",
    price: "₹250 for two",
    priceNum: 250,
    distance: "1.5 km",
    image: "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&q=80",
    tags: ["Under ₹300"],
    offer: "Buy 1 Get 1 FREE",
    isVeg: false,
    featured: true,
    category: "Burger",
    menu: [
      {
        section: "Signature Burgers",
        items: [
          { id: 301, name: "Classic Cheeseburger", desc: "Beef patty, American cheese, pickles, mustard, ketchup", price: 199, rating: 4.3, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=300&q=80" },
          { id: 302, name: "Double Smash Burger", desc: "Two smashed beef patties, special sauce, caramelized onions", price: 279, rating: 4.6, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1553979459-d2229ba7433b?w=300&q=80" },
          { id: 303, name: "Crispy Chicken Burger", desc: "Fried chicken breast, coleslaw, honey mustard", price: 229, rating: 4.4, veg: false, image: "https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=300&q=80" },
          { id: 304, name: "Mushroom Swiss", desc: "Sautéed mushrooms, Swiss cheese, garlic mayo", price: 249, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1586190848861-99aa4a171e90?w=300&q=80" },
        ]
      },
      {
        section: "Sides",
        items: [
          { id: 305, name: "Loaded Fries", desc: "Crispy fries with cheese sauce, jalapeños, sour cream", price: 149, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=300&q=80" },
          { id: 306, name: "Onion Rings", desc: "Beer-battered crispy onion rings with dipping sauce", price: 99, rating: 4.1, veg: true, image: "https://images.unsplash.com/photo-1639024471283-03518883512d?w=300&q=80" },
          { id: 310, name: "Coke (500ml)", desc: "Chilled Coca-Cola soda", price: 49, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1572558116176-c31e5e619a43?w=300&q=80" },
        ]
      },
      {
        section: "Shakes",
        items: [
          { id: 307, name: "Nutella Milkshake", desc: "Thick creamy shake with Nutella and milk", price: 149, rating: 4.5, veg: true, image: "https://images.unsplash.com/photo-1572490122747-3968b75cc699?w=300&q=80" },
          { id: 308, name: "Oreo Blizzard", desc: "Vanilla soft serve with crushed Oreos", price: 129, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1534353473418-4cfa0669e93d?w=300&q=80" },
        ]
      }
    ]
  },
  {
    id: 4,
    name: "Sushi Zen",
    cuisine: "Japanese • Sushi • Asian",
    rating: 4.7,
    reviews: 876,
    time: "40-50 min",
    price: "₹700 for two",
    priceNum: 700,
    distance: "2.1 km",
    image: "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400&q=80",
    tags: ["Premium", "Top Rated"],
    offer: "Free miso soup on orders above ₹800",
    isVeg: false,
    featured: false,
    category: "Sushi",
    menu: [
      {
        section: "Appetizers",
        items: [
          { id: 399, name: "Steamed Momos (6 pcs)", desc: "Assorted veg and chicken momos served with spicy chilli sauce", price: 199, rating: 4.4, veg: false, image: "https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=300&q=80" },
        ]
      },
      {
        section: "Sushi Rolls",
        items: [
          { id: 401, name: "Dragon Roll", desc: "Shrimp tempura, cucumber topped with avocado, eel sauce", price: 480, rating: 4.7, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1617196034183-421b4040ed20?w=300&q=80" },
          { id: 402, name: "Spicy Tuna Roll", desc: "Fresh tuna, spicy mayo, cucumber, sesame seeds", price: 420, rating: 4.6, veg: false, image: "https://images.unsplash.com/photo-1559410545-0bdcd187e0a6?w=300&q=80" },
          { id: 403, name: "Vegetable Roll", desc: "Cucumber, avocado, carrot, pickled radish", price: 280, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1562802378-063ec186a863?w=300&q=80" },
        ]
      },
      {
        section: "Nigiri & Sashimi",
        items: [
          { id: 404, name: "Salmon Nigiri (2 pcs)", desc: "Fresh Atlantic salmon over seasoned rice", price: 320, rating: 4.8, veg: false, bestseller: true, image: "https://images.unsplash.com/photo-1617196034183-421b4040ed20?w=300&q=80" },
          { id: 405, name: "Tuna Sashimi (5 pcs)", desc: "Premium-grade tuna, sliced thin", price: 450, rating: 4.7, veg: false, image: "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=300&q=80" },
        ]
      },
      {
        section: "Mains",
        items: [
          { id: 406, name: "Chicken Ramen", desc: "Rich tonkotsu broth, chashu pork, soft boiled egg, nori", price: 380, rating: 4.5, veg: false, image: "https://images.unsplash.com/photo-1569050467447-ce54b3bbc37d?w=300&q=80" },
          { id: 407, name: "Edamame", desc: "Steamed salted soybean pods", price: 120, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1553361371-9b22f78e8b1d?w=300&q=80" },
        ]
      }
    ]
  },
  {
    id: 5,
    name: "The Green Bowl",
    cuisine: "Salads • Healthy • Wraps",
    rating: 4.4,
    reviews: 1234,
    time: "20-25 min",
    price: "₹350 for two",
    priceNum: 350,
    distance: "0.5 km",
    image: "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&q=80",
    tags: ["Healthy", "Under 500 cal"],
    offer: "Free protein add-on today",
    isVeg: true,
    featured: false,
    category: "Salads",
    menu: [
      {
        section: "Power Bowls",
        items: [
          { id: 501, name: "Mediterranean Bowl", desc: "Quinoa, falafel, hummus, cucumber, olives, feta", price: 299, rating: 4.5, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=300&q=80" },
          { id: 502, name: "Protein Beast Bowl", desc: "Grilled chicken, sweet potato, kale, tahini dressing", price: 349, rating: 4.4, veg: false, image: "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=300&q=80" },
          { id: 503, name: "Acai Bowl", desc: "Acai blend topped with granola, banana, berries, honey", price: 279, rating: 4.6, veg: true, image: "https://images.unsplash.com/photo-1547592180-85f173990554?w=300&q=80" },
        ]
      },
      {
        section: "Salads",
        items: [
          { id: 504, name: "Caesar Salad", desc: "Romaine, parmesan, croutons, classic Caesar dressing", price: 229, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=300&q=80" },
          { id: 505, name: "Greek Salad", desc: "Tomatoes, cucumber, olives, feta, red onion, oregano", price: 249, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1595670002930-b30d563cf121?w=300&q=80" },
        ]
      },
      {
        section: "Wraps & Juices",
        items: [
          { id: 506, name: "Grilled Veggie Wrap", desc: "Zucchini, bell pepper, hummus, spinach in whole wheat wrap", price: 199, rating: 4.2, veg: true, image: "https://images.unsplash.com/photo-1626700051175-6818013e1d4f?w=300&q=80" },
          { id: 507, name: "Green Detox Juice", desc: "Spinach, cucumber, celery, lemon, ginger, apple", price: 149, rating: 4.5, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1610970881699-44a5587cabec?w=300&q=80" },
        ]
      }
    ]
  },
  {
    id: 6,
    name: "Cake & Co.",
    cuisine: "Bakery • Desserts • Coffee",
    rating: 4.6,
    reviews: 2107,
    time: "15-25 min",
    price: "₹200 for two",
    priceNum: 200,
    distance: "0.3 km",
    image: "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=400&q=80",
    tags: ["Trending", "Desserts"],
    offer: "10% OFF on orders above ₹500",
    isVeg: true,
    featured: true,
    category: "Desserts",
    menu: [
      {
        section: "Cakes",
        items: [
          { id: 601, name: "Belgian Chocolate Cake", desc: "Rich dark chocolate layers with ganache frosting", price: 320, rating: 4.7, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=300&q=80" },
          { id: 602, name: "Red Velvet Cake", desc: "Classic red velvet with cream cheese frosting", price: 290, rating: 4.6, veg: true, bestseller: true, image: "https://images.unsplash.com/photo-1535141192574-5d4897c12636?w=300&q=80" },
          { id: 603, name: "Tiramisu", desc: "Classic Italian dessert with espresso and mascarpone", price: 180, rating: 4.5, veg: true, image: "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=300&q=80" },
        ]
      },
      {
        section: "Pastries & Cookies",
        items: [
          { id: 604, name: "Croissant", desc: "Buttery, flaky French croissant", price: 80, rating: 4.3, veg: true, image: "https://images.unsplash.com/photo-1555507036-ab1f4038808a?w=300&q=80" },
          { id: 605, name: "Chocolate Chip Cookies (4 pcs)", desc: "Fresh baked with semi-sweet chocolate chips", price: 120, rating: 4.4, veg: true, image: "https://images.unsplash.com/photo-1499636136210-6f4ee915583e?w=300&q=80" },
          { id: 606, name: "Cinnamon Roll", desc: "Warm, gooey cinnamon roll with cream cheese glaze", price: 110, rating: 4.5, veg: true, image: "https://images.unsplash.com/photo-1509365465985-25d11c17e812?w=300&q=80" },
          { id: 609, name: "Fudgy Brownie", desc: "Rich chocolate brownie with nuts", price: 130, rating: 4.6, veg: true, image: "https://images.unsplash.com/photo-1604335399104-d7f15b90b639?w=300&q=80" },
        ]
      },
      {
        section: "Coffee & Beverages",
        items: [
          { id: 607, name: "Cappuccino", desc: "Espresso with steamed milk foam, light & frothy", price: 120, rating: 4.4, veg: true, bestseller: true, image: "https://www.allrecipes.com/thmb/BmpkR1pEH4WBzHgoH-oMQloHZi8=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/8624835-how-to-make-a-cappuccino-beauty-4x3-0301-13d55eaad60b42058f24369c292d4ccb.jpg" },
          { id: 608, name: "Iced Caramel Latte", desc: "Cold brew coffee with caramel syrup and milk", price: 150, rating: 4.6, veg: true, image: "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?w=300&q=80" },
        ]
      }
    ]
  },
];

// ===== SAVED ADDRESSES =====
const SAVED_ADDRESSES = [
  {
    id: 1,
    type: "Home",
    name: "Rahul Sharma",
    phone: "+91 98765 43210",
    house: "Flat 402, Sunshine Apartments",
    area: "Linking Road, Bandra West",
    city: "Mumbai",
    pin: "400050"
  },
  {
    id: 2,
    type: "Work",
    name: "Rahul Sharma",
    phone: "+91 98765 43210",
    house: "12th Floor, Tech Park",
    area: "BKC, Bandra Kurla Complex",
    city: "Mumbai",
    pin: "400051"
  }
];

// ===== OFFERS =====
const COUPONS = [
  { code: "NEWBIE40", discount: 40, type: "percent", maxDiscount: 80, minOrder: 0, label: "40% OFF upto ₹80" },
  { code: "FLAT100", discount: 100, type: "flat", maxDiscount: 100, minOrder: 200, label: "Flat ₹100 OFF" },
  { code: "FREEDEL", discount: 30, type: "flat", maxDiscount: 30, minOrder: 0, label: "Free Delivery (saves ₹30)" },
];
