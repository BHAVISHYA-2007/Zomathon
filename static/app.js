// ===== STATE =====
let cart = [];
let currentRestaurant = null;
let selectedAddress = SAVED_ADDRESSES[0];
let selectedPayment = 'upi';
let appliedCoupon = null;
let pageHistory = ['home'];
let orderDetails = null;
let trackingInterval = null;

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
  renderCategories();
  renderTopPicks();
  renderRecommended();
  renderCuisines();
  updateCartUI();
  setupAddressTypeBtns();
  setupNavSearch();
  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 10);
    // scroll-to-top button
    const btn = document.getElementById('scrollTopBtn');
    if (window.scrollY > 300) {
      btn.style.display = 'block';
    } else {
      btn.style.display = 'none';
    }
  });

  // Dark mode toggle listener
  const darkBtn = document.getElementById('darkToggle');
  if (darkBtn) {
    darkBtn.addEventListener('click', () => toggleDarkMode());
  }

  // Apply saved dark mode preference
  if (localStorage.getItem('darkMode') === 'true') toggleDarkMode(true);

});

// ===== PAGE NAVIGATION =====
function showPage(page, data) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const target = document.getElementById('page-' + page);
  if (!target) return;
  target.classList.add('active');
  window.scrollTo(0, 0);

  if (pageHistory[pageHistory.length - 1] !== page) {
    pageHistory.push(page);
  }

  if (page === 'restaurant' && data) {
    currentRestaurant = data;
    renderRestaurantPage(data);
  }
  if (page === 'cart') renderCartPage();
  if (page === 'address') renderAddressPage();
  if (page === 'payment') renderPaymentPage();
  if (page === 'search' && data) renderSearchPage(data);
}

function goBack() {
  pageHistory.pop();
  const prev = pageHistory[pageHistory.length - 1] || 'home';
  showPage(prev);
}

// ===== CATEGORIES =====
// helper to add/remove classes for gradient overlays based on scroll position
function updateScrollableGradient(gridId) {
  const grid = document.getElementById(gridId);
  if (!grid) return;
  const container = grid.parentElement; // should be categories-container or cuisine-container
  const check = () => {
    const atStart = grid.scrollLeft <= 5;
    const atEnd = grid.scrollLeft + grid.clientWidth >= grid.scrollWidth - 5;
    container.classList.toggle('scrolled-start', !atStart);
    container.classList.toggle('scrolled-end', !atEnd);
  };
  grid.addEventListener('scroll', check);
  // run once
  setTimeout(check, 0);
}

function renderCategories() {
  const grid = document.getElementById('categoriesGrid');
  grid.innerHTML = CATEGORIES.map(c => `
    <div class="category-card" onclick="filterByCategory('${c.name}')"
      style="--cat-color:${c.color}">
      <div class="cat-emoji-wrap" style="background:${c.color}22">
        <span class="cat-emoji">${c.emoji}</span>
      </div>
      <div class="cat-name">${c.name}</div>
    </div>
  `).join('');
  updateScrollableGradient('categoriesGrid');
}

function filterByCategory(name) {
  const results = RESTAURANTS.filter(r => r.category === name || r.cuisine.includes(name));
  showPage('search', { query: name, results });
}

// ===== NAVBAR HELPERS =====
function toggleNavMenu() {
  const container = document.querySelector('.nav-container');
  container.classList.toggle('mobile-open');
}

function toggleDarkMode(force) {
  const root = document.documentElement;
  const enabled = force !== undefined ? force : !root.classList.contains('dark');
  if (enabled) {
    root.classList.add('dark');
    localStorage.setItem('darkMode', 'true');
    document.getElementById('darkToggle').innerHTML = '<i class="fas fa-sun"></i>';
  } else {
    root.classList.remove('dark');
    localStorage.setItem('darkMode', 'false');
    document.getElementById('darkToggle').innerHTML = '<i class="fas fa-moon"></i>';
  }
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===== ANIMATIONS =====
function flyToCart(img) {
  const cartBtn = document.querySelector('.btn-cart');
  if (!cartBtn) return;
  const clone = img.cloneNode(true);
  const rect = img.getBoundingClientRect();
  clone.style.position = 'fixed';
  clone.style.left = rect.left + 'px';
  clone.style.top = rect.top + 'px';
  clone.style.width = rect.width + 'px';
  clone.style.height = rect.height + 'px';
  clone.style.transition = 'all 0.8s ease-in-out';
  clone.style.zIndex = 1000;
  document.body.appendChild(clone);
  const cartRect = cartBtn.getBoundingClientRect();
  setTimeout(() => {
    clone.style.left = cartRect.left + 'px';
    clone.style.top = cartRect.top + 'px';
    clone.style.width = '20px';
    clone.style.height = '20px';
    clone.style.opacity = '0.5';
  }, 20);
  setTimeout(() => clone.remove(), 900);
}

// ===== RESTAURANT CARDS =====
function restaurantCard(r) {
  return `
    <div class="restaurant-card" onclick="showPage('restaurant', RESTAURANTS.find(x=>x.id===${r.id}))">
      <div class="card-img-wrap">
        <img src="${r.image}" alt="${r.name}" loading="lazy">
        ${r.offer ? `<div class="card-offer"><i class="fas fa-tag"></i> ${r.offer}</div>` : ''}
        ${r.tags.map(t => `<span class="card-tag">${t}</span>`).join('')}
        <button class="card-wishlist" onclick="event.stopPropagation(); this.classList.toggle('liked'); this.textContent=this.classList.contains('liked')?'❤️':'🤍'">🤍</button>
      </div>
      <div class="card-body">
        <div class="card-top">
          <h3 class="card-name">${r.name}</h3>
          <div class="card-rating"><i class="fas fa-star"></i> ${r.rating}</div>
        </div>
        <p class="card-cuisine">${r.cuisine}</p>
        <div class="card-meta">
          <i class="fas fa-clock"></i><span>${r.time}</span>
          <span class="dot">•</span>
          <span>${r.price}</span>
          <span class="dot">•</span>
          <i class="fas fa-map-marker-alt"></i><span>${r.distance}</span>
        </div>
      </div>
    </div>
  `;
}

function renderTopPicks() {
  const grid = document.getElementById('topPicksGrid');
  const filters = document.getElementById('filters');
  const filterOptions = ['All', 'Pure Veg', 'Under ₹300', 'Rating 4.5+', 'Fastest Delivery'];
  filters.innerHTML = filterOptions.map((f, i) =>
    `<button class="filter-btn ${i === 0 ? 'active' : ''}" onclick="applyFilter('${f}', this)">${f}</button>`
  ).join('');
  grid.innerHTML = RESTAURANTS.map(restaurantCard).join('');
}

function applyFilter(filter, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  let filtered = [...RESTAURANTS];
  if (filter === 'Pure Veg') filtered = filtered.filter(r => r.isVeg);
  else if (filter === 'Under ₹300') filtered = filtered.filter(r => r.priceNum <= 300);
  else if (filter === 'Rating 4.5+') filtered = filtered.filter(r => r.rating >= 4.5);
  else if (filter === 'Fastest Delivery') filtered = filtered.sort((a, b) => parseInt(a.time) - parseInt(b.time));
  document.getElementById('topPicksGrid').innerHTML = filtered.map(restaurantCard).join('');
}

function renderRecommended() {
  const shuffled = [...RESTAURANTS].sort(() => 0.5 - Math.random()).slice(0, 4);
  document.getElementById('recommendedGrid').innerHTML = shuffled.map(restaurantCard).join('');
}

function renderCuisines() {
  const grid = document.getElementById('cuisineGrid');
  grid.innerHTML = CUISINES.map(c => `
    <div class="cuisine-card" style="--c:${c.color}" onclick="filterByCategory('${c.name}')">
      <div class="cuisine-emoji">${c.emoji}</div>
      <div class="cuisine-info">
        <div class="cuisine-name">${c.name}</div>
        <div class="cuisine-count">${c.count}</div>
      </div>
      <i class="fas fa-arrow-right cuisine-arrow"></i>
    </div>
  `).join('');
  updateScrollableGradient('cuisineGrid');
}

// ===== RESTAURANT PAGE =====
function renderRestaurantPage(r) {
  // Hero
  document.getElementById('restaurantHero').innerHTML = `
    <div class="rest-hero-bg" style="background-image:url('${r.image}')"></div>
    <div class="rest-hero-overlay">
      <button class="back-btn white" onclick="goBack()"><i class="fas fa-arrow-left"></i> Back</button>
      <div class="rest-hero-info">
        <h1 class="rest-name">${r.name}</h1>
        <p class="rest-cuisine">${r.cuisine}</p>
        <div class="rest-badges">
          <span class="rest-badge"><i class="fas fa-star"></i> ${r.rating} (${r.reviews.toLocaleString()} reviews)</span>
          <span class="rest-badge"><i class="fas fa-clock"></i> ${r.time}</span>
          <span class="rest-badge"><i class="fas fa-map-marker-alt"></i> ${r.distance}</span>
          <span class="rest-badge"><i class="fas fa-rupee-sign"></i> ${r.price}</span>
        </div>
        ${r.offer ? `<div class="rest-offer"><i class="fas fa-tag"></i> ${r.offer}</div>` : ''}
      </div>
    </div>
  `;

  // Sticky nav
  const sections = r.menu.map(s => s.section);
  document.getElementById('restaurantStickyNav').innerHTML = `
    <div class="sticky-nav-inner">
      ${sections.map((s, i) => `<a class="sticky-link ${i === 0 ? 'active' : ''}" onclick="scrollToSection('${s}', this)">${s}</a>`).join('')}
      <button class="sticky-veg-toggle ${r.isVeg ? 'on' : ''}" onclick="toggleVeg(this)">
        <span class="veg-icon">🥦</span> Veg Only
      </button>
    </div>
  `;

  // Sidebar
  document.getElementById('menuSidebar').innerHTML = `
    <div class="sidebar-inner">
      ${sections.map(s => `<div class="sidebar-item" onclick="scrollToSection('${s}')">${s}</div>`).join('')}
    </div>
  `;

  // Menu main
  document.getElementById('menuMain').innerHTML = `
    <div class="menu-search">
      <i class="fas fa-search"></i>
      <input type="text" placeholder="Search in menu" oninput="searchMenu(this.value, ${r.id})">
    </div>
    ${r.menu.map(section => `
      <div class="menu-section" id="section-${section.section.replace(/\s+/g, '-')}">
        <h3 class="menu-section-title">${section.section} <span>(${section.items.length})</span></h3>
        ${section.items.map(item => menuItemCard(item, r.id)).join('')}
      </div>
    `).join('')}
  `;
}

function menuItemCard(item, restId) {
  const inCart = cart.find(c => c.id === item.id);
  const qty = inCart ? inCart.qty : 0;
  return `
    <div class="menu-item" id="item-${item.id}">
      <div class="menu-item-info">
        <div class="item-veg-icon ${item.veg ? 'veg' : 'non-veg'}">
          <span></span>
        </div>
        ${item.bestseller ? '<span class="bestseller-badge">⭐ Bestseller</span>' : ''}
        <h4 class="item-name">${item.name}</h4>
        <div class="item-rating"><i class="fas fa-star"></i> ${item.rating}</div>
        <p class="item-price">₹${item.price}</p>
        <p class="item-desc">${item.desc}</p>
      </div>
      <div class="menu-item-right">
        ${item.image ? `<img src="${item.image}" alt="${item.name}" class="item-img" loading="lazy">` : ''}
        <div class="qty-control" id="qty-${item.id}">
          ${qty === 0
      ? `<button class="add-btn" onclick="addToCart(${item.id}, ${restId})">+ ADD</button>`
      : `<div class="qty-stepper">
                <button onclick="decreaseQty(${item.id})">−</button>
                <span>${qty}</span>
                <button onclick="increaseQty(${item.id}, ${restId})">+</button>
               </div>`
    }
        </div>
      </div>
    </div>
  `;
}

function scrollToSection(sectionName, linkEl) {
  const id = 'section-' + sectionName.replace(/\s+/g, '-');
  const el = document.getElementById(id);
  if (el) {
    const offset = el.getBoundingClientRect().top + window.scrollY - 160;
    window.scrollTo({ top: offset, behavior: 'smooth' });
  }
  if (linkEl) {
    document.querySelectorAll('.sticky-link').forEach(l => l.classList.remove('active'));
    linkEl.classList.add('active');
  }
}

let vegOnly = false;
function toggleVeg(btn) {
  vegOnly = !vegOnly;
  btn.classList.toggle('on', vegOnly);
  if (!currentRestaurant) return;
  currentRestaurant.menu.forEach(section => {
    section.items.forEach(item => {
      const el = document.getElementById('item-' + item.id);
      if (el) el.style.display = (vegOnly && !item.veg) ? 'none' : 'flex';
    });
  });
}

function searchMenu(query, restId) {
  const rest = RESTAURANTS.find(r => r.id === restId);
  if (!rest) return;
  rest.menu.forEach(section => {
    section.items.forEach(item => {
      const el = document.getElementById('item-' + item.id);
      if (el) {
        const match = item.name.toLowerCase().includes(query.toLowerCase()) || item.desc.toLowerCase().includes(query.toLowerCase());
        el.style.display = (query === '' || match) ? 'flex' : 'none';
      }
    });
  });
}

// ===== CART =====
function getItemById(id) {
  for (const r of RESTAURANTS) {
    for (const section of r.menu) {
      const item = section.items.find(i => i.id === id);
      if (item) return { item, restaurant: r };
    }
  }
  return null;
}

function addToCart(itemId, restId) {
  const data = getItemById(itemId);
  if (!data) return;

  // animation using image element
  const cardImg = document.querySelector(`#item-${itemId} .item-img`);
  if (cardImg) flyToCart(cardImg);

  if (cart.length > 0 && cart[0].restaurantId !== restId) {
    if (!confirm(`Your cart has items from ${cart[0].restaurantName}. Clear cart and add from ${data.restaurant.name}?`)) return;
    cart = [];
  }

  const existing = cart.find(c => c.id === itemId);
  if (existing) {
    existing.qty++;
  } else {
    cart.push({
      id: itemId,
      name: data.item.name,
      price: data.item.price,
      qty: 1,
      restaurantId: restId,
      restaurantName: data.restaurant.name,
      image: data.item.image,
      veg: data.item.veg
    });
  }

  updateCartUI();
  updateQtyControl(itemId);
  showToast(`${data.item.name} added to cart!`);
}

function increaseQty(itemId, restId) {
  const existing = cart.find(c => c.id === itemId);
  if (existing) {
    existing.qty++;
  } else {
    addToCart(itemId, restId);
    return;
  }
  updateCartUI();
  updateQtyControl(itemId);
}

function decreaseQty(itemId) {
  const idx = cart.findIndex(c => c.id === itemId);
  if (idx === -1) return;
  cart[idx].qty--;
  if (cart[idx].qty <= 0) cart.splice(idx, 1);
  updateCartUI();
  updateQtyControl(itemId);
}

function updateQtyControl(itemId) {
  const el = document.getElementById('qty-' + itemId);
  if (!el) return;
  const item = cart.find(c => c.id === itemId);
  const qty = item ? item.qty : 0;
  el.innerHTML = qty === 0
    ? `<button class="add-btn" onclick="addToCart(${itemId}, ${currentRestaurant ? currentRestaurant.id : 0})">+ ADD</button>`
    : `<div class="qty-stepper">
        <button onclick="decreaseQty(${itemId})">−</button>
        <span>${qty}</span>
        <button onclick="increaseQty(${itemId}, ${currentRestaurant ? currentRestaurant.id : 0})">+</button>
       </div>`;
}

function getCartTotal() {
  return cart.reduce((sum, item) => sum + item.price * item.qty, 0);
}

function getCartCount() {
  return cart.reduce((sum, item) => sum + item.qty, 0);
}

function updateCartUI() {
  const count = getCartCount();
  document.getElementById('cartCount').textContent = count;

  const miniCart = document.getElementById('miniCart');
  if (count > 0 && document.querySelector('.page.active')?.id !== 'page-cart') {
    const total = getCartTotal();
    miniCart.innerHTML = `
      <div class="mini-cart-inner">
        <span>${count} item${count > 1 ? 's' : ''} | ₹${total}</span>
        <span>View Cart <i class="fas fa-arrow-right"></i></span>
      </div>
    `;
    miniCart.classList.add('visible');
  } else {
    miniCart.classList.remove('visible');
  }
}

// ===== CART PAGE =====
function renderCartPage() {
  updateCartUI();
  const miniCart = document.getElementById('miniCart');
  miniCart.classList.remove('visible');

  if (cart.length === 0) {
    document.getElementById('cartItemsWrap').innerHTML = `
      <div class="empty-state" style="border:none; box-shadow:none;">
        <div class="empty-emoji">🛒</div>
        <h3>Your cart is empty</h3>
        <p>Add items from a restaurant to get started</p>
        <button class="btn-primary" onclick="showPage('home')">Browse Restaurants</button>
      </div>
    `;
    document.getElementById('cartRestaurantInfo').innerHTML = '';
    document.getElementById('addMoreCartBtn').style.display = 'none';
    document.querySelector('.combo-section').style.display = 'none';
    document.querySelector('.ai-section').style.display = 'none';

    // reset bill text
    document.getElementById('billSubtotal').textContent = '₹0';
    document.getElementById('billTotal').textContent = '₹0';
    document.getElementById('cartChip').textContent = '0 items';
    return;
  }

  document.getElementById('addMoreCartBtn').style.display = 'flex';
  document.querySelector('.combo-section').style.display = 'block';
  document.querySelector('.ai-section').style.display = 'block';

  const r = RESTAURANTS.find(r => r.id === cart[0].restaurantId);
  document.getElementById('cartRestaurantInfo').innerHTML = `
    <div class="cart-restaurant">
      <img src="${r?.image || ''}" alt="${r?.name}">
      <div>
        <h3>${r?.name}</h3>
        <p>${r?.cuisine}</p>
      </div>
    </div>
  `;

  document.getElementById('addMoreCartBtn').onclick = () => showPage('restaurant', r);

  document.getElementById('cartItemsWrap').innerHTML = cart.map((item, i) => `
    <div class="cart-item" style="animation-delay:${i * 0.05}s">
      <div class="cart-item-veg ${item.veg ? 'veg' : 'non-veg'}"><span></span></div>
      ${item.image ? `<img src="${item.image}" style="width:48px;height:48px;border-radius:8px;object-fit:cover;flex-shrink:0;box-shadow:var(--shadow-sm);margin-right:12px;">` : ''}
      <div class="cart-item-info">
        <span class="cart-item-name">${item.name}</span>
        <div style="font-size:12px;color:var(--text-hint);margin-top:2px">${item.desc || 'From ' + r.name}</div>
        <span class="custom-tag" onclick="alert('Customisation coming soon!')" style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--sage);background:var(--sage-light);padding:3px 8px;border-radius:var(--radius-pill);margin-top:6px;cursor:pointer;font-weight:700;"><i class="fas fa-sliders-h"></i> Customise</span>
      </div>
      <div class="cart-item-controls">
        <div class="qty-stepper">
          <button onclick="decreaseQty(${item.id}); renderCartPage()">−</button>
          <span>${item.qty}</span>
          <button onclick="increaseQty(${item.id}, ${cart[0].restaurantId}); renderCartPage()">+</button>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
            <span class="cart-item-price">₹${item.price * item.qty}</span>
            <button class="item-del" onclick="removeAllQty(${item.id})" title="Remove" style="background:none;border:none;color:var(--text-hint);cursor:pointer;font-size:13px;"><i class="fas fa-trash-alt"></i></button>
        </div>
      </div>
    </div>
  `).join('');

  const totalQty = cart.reduce((s, i) => s + i.qty, 0);
  document.getElementById('cartChip').textContent = totalQty + ' item' + (totalQty !== 1 ? 's' : '');

  // Render Smart features
  renderRecs('all');
  renderCombo();
  renderCouponChips();
  renderCartSummary();
}

function removeAllQty(id) {
  cart = cart.filter(i => i.id !== id);
  updateCartUI();
  renderCartPage();
}

function renderCartSummary() {
  const subtotal = getCartTotal();
  const delivery = 30;
  const platformFee = 5;
  let discount = 0;
  if (appliedCoupon) {
    if (appliedCoupon.type === 'percent') {
      discount = Math.min(Math.round(subtotal * appliedCoupon.discount / 100), appliedCoupon.maxDiscount);
    } else {
      discount = appliedCoupon.discount;
    }
  }
  const total = subtotal + delivery + platformFee - discount;

  document.getElementById('billSubtotal').textContent = '₹' + subtotal;
  document.getElementById('billTotal').textContent = '₹' + total;

  const discRow = document.getElementById('discountRow');
  const discountAmtEl = document.getElementById('discountAmt');
  if (discount > 0 && discRow) {
    discRow.style.display = 'flex';
    if (discountAmtEl) discountAmtEl.textContent = '– ₹' + discount;
    const savPill = document.getElementById('savingsPill');
    if (savPill) {
      savPill.style.display = 'flex';
      savPill.innerHTML = '<i class="fas fa-piggy-bank"></i> Saved ₹' + discount;
    }
  } else if (discRow) {
    discRow.style.display = 'none';
    const savPill = document.getElementById('savingsPill');
    if (savPill) savPill.style.display = 'none';
  }
}

function applyCoupon() {
  const code = document.getElementById('couponInput')?.value.trim().toUpperCase();
  const coupon = COUPONS.find(c => c.code === code);
  const msg = document.getElementById('couponMsg');
  if (coupon) {
    appliedCoupon = coupon;
    const saveAmt = coupon.type === 'flat' ? coupon.discount : Math.min(Math.round(getCartTotal() * coupon.discount / 100), coupon.maxDiscount);
    msg.innerHTML = `<div class="coupon-success"><i class="fas fa-check-circle"></i> "${coupon.code}" applied! You save ₹${saveAmt} 🎉</div>`;
    renderCartSummary();
  } else {
    msg.innerHTML = `<div class="coupon-error"><i class="fas fa-times-circle"></i> Invalid coupon code. Try NEWBIE40</div>`;
    appliedCoupon = null;
    renderCartSummary();
  }
}

// ─── SMART CART LOGIC ───
let addedRecs = new Set();
let moreLoaded = false;
let currentFilter = 'all';

async function renderRecs(filter = 'all') {
  const container = document.getElementById('aiRecs');
  if (!container) return;
  const restaurantId = cart.length ? cart[0].restaurantId : null;
  const currentRest = RESTAURANTS.find(r => r.id === restaurantId);

  if (!currentRest) return;

  // Show loading state
  container.innerHTML = '<div style="text-align:center;padding:20px;"><i class="fas fa-spinner fa-spin"></i> AI analyzing your preferences...</div>';

  let recs = [];

  try {
    // Get current cart items
    const cartItems = cart.map(item => {
      const foundItem = currentRest.menu.flatMap(s => s.items).find(i => i.id === item.id);
      return foundItem?.name || 'item';
    });

    // Call API endpoint
    const response = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: cartItems })
    });

    const data = await response.json();
    
    if (data.success && data.recommendations && data.recommendations.length > 0) {
      // Map recommendations to items
      data.recommendations.forEach((recName, idx) => {
        currentRest.menu.forEach(section => {
          section.items.forEach(item => {
            if (recs.length < 5 && item.name.toLowerCase().includes(recName.toLowerCase())) {
              recs.push({
                id: item.id,
                type: idx % 3 === 0 ? 'popular' : (idx % 2 === 0 ? 'pair' : 'ai'),
                name: item.name,
                price: item.price,
                desc: item.desc || item.name,
                img: item.image || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=200&q=80',
                rating: item.rating,
                cal: (200 + idx * 50) + ' kcal',
                confidence: 90 + (Math.random() * 9),
                reason: ['🤖 AI Pick', '🍷 Perfect pairing', '✨ Smart Match'][idx % 3],
                reasonType: ['ai-type', 'pair-type', 'ai-type'][idx % 3],
                veg: item.veg
              });
            }
          });
        });
      });
    }

    // Fallback to local recommendations if API didn't return enough
    if (recs.length < 5) {
      currentRest.menu.forEach(section => {
        section.items.forEach((item, idx) => {
          if (recs.length < 5 && idx % 2 === 0 && !recs.some(r => r.id === item.id)) {
            recs.push({
              id: item.id,
              type: idx % 3 === 0 ? 'popular' : (idx % 2 === 0 ? 'pair' : 'ai'),
              name: item.name,
              price: item.price,
              desc: item.desc || item.name,
              img: item.image || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=200&q=80',
              rating: item.rating,
              cal: (200 + idx * 50) + ' kcal',
              confidence: 75 + idx,
              reason: ['🔥 Trending', '🍷 Perfect pairing', '✨ Popular'][idx % 3],
              reasonType: ['popular-type', 'pair-type', 'ai-type'][idx % 3],
              veg: item.veg
            });
          }
        });
      });
    }
  } catch (error) {
    console.warn('API error, using local recommendations:', error);
    // Fallback to local recommendations
    currentRest.menu.forEach(section => {
      section.items.forEach((item, idx) => {
        if (recs.length < 5 && idx % 2 === 0) {
          recs.push({
            id: item.id,
            type: idx % 3 === 0 ? 'popular' : (idx % 2 === 0 ? 'pair' : 'ai'),
            name: item.name,
            price: item.price,
            desc: item.desc || item.name,
            img: item.image || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=200&q=80',
            rating: item.rating,
            cal: (200 + idx * 50) + ' kcal',
            confidence: 85 + idx,
            reason: ['🔥 Trending', '🍷 Perfect pairing', '✨ AI Pick'][idx % 3],
            reasonType: ['popular-type', 'pair-type', 'ai-type'][idx % 3],
            veg: item.veg
          });
        }
      });
    });
  }

  if (filter !== 'all') {
    recs = recs.filter(r => r.type === filter);
  }

  container.innerHTML = recs.map(r => {
    const isAdded = cart.some(c => c.id === r.id);
    return `
    <div class="rec-card ${r.type === 'popular' ? 'hot' : ''}" id="rec-${r.id}">
      <div class="rec-reason ${r.reasonType}"><i class="fas fa-robot"></i> ${r.reason}</div>
      <div class="rec-body">
        <img class="rec-img" src="${r.img}" alt="${r.name}" loading="lazy">
        <div class="rec-info">
          <div class="rec-name">${r.name}</div>
          <div class="rec-desc">${r.desc}</div>
          <div class="rec-meta">
            <span class="rec-price">₹${r.price}</span>
            <span class="rec-rating"><i class="fas fa-star"></i> ${r.rating}</span>
            <span class="rec-cal">${r.cal}</span>
          </div>
        </div>
        <div class="rec-right">
          <div class="ai-confidence">
            <span class="conf-label">AI ${Math.round(r.confidence)}%</span>
            <div class="conf-bar"><div class="conf-fill" style="width:${r.confidence}%"></div></div>
          </div>
          <button class="rec-add-btn ${r.type === 'popular' ? 'flame' : ''} ${isAdded ? 'added' : ''}"
            onclick="addRec(${r.id}, '${r.name.replace(/'/g, "\\'")}', ${r.price}, ${r.veg}, '${r.img}')">
            ${isAdded ? '<i class="fas fa-check"></i> Added' : '+ Add'}
          </button>
        </div>
      </div>
    </div>`;
  }).join('');

  // Nudge logic
  if (recs.length > 0) {
    document.getElementById('nudgeItemName').textContent = recs[0].name;
    document.getElementById('nudgeSaveAmount').textContent = 'Save ₹20';
  }
}

function addRec(id, name, price, veg, img) {
  const restaurantId = cart.length ? cart[0].restaurantId : null;
  addToCart(id, restaurantId);
  renderCartPage();
}

function filterRecs(type, btn) {
  document.querySelectorAll('.ai-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentFilter = type;
  renderRecs(type);
}

function loadMoreRecs() {
  const btn = document.getElementById('seeMoreBtn');
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> AI is loading more...';
  setTimeout(() => {
    btn.innerHTML = '<i class="fas fa-check"></i> All recommendations loaded';
    btn.style.color = 'var(--sage)';
  }, 1000);
}

function addFromNudge() {
  const restaurantId = cart.length ? cart[0].restaurantId : null;
  const currentRest = RESTAURANTS.find(r => r.id === restaurantId);
  if (currentRest && currentRest.menu[0].items[0]) {
    addToCart(currentRest.menu[0].items[0].id, restaurantId);
    const nudgeBtn = document.getElementById('nudgeCtaBtn');
    if (nudgeBtn) {
      document.getElementById('aiNudge').style.opacity = '0.5';
      nudgeBtn.textContent = '✓ Added!';
    }
    renderCartPage();
  }
}

function renderCombo() {
  const wrap = document.getElementById('comboItems');
  if (!wrap) return;
  const restaurantId = cart.length ? cart[0].restaurantId : null;
  const currentRest = RESTAURANTS.find(r => r.id === restaurantId);

  if (!currentRest) return;

  let comboItems = [];
  currentRest.menu.forEach(section => {
    if (comboItems.length < 4 && section.items.length > 0) {
      comboItems.push({ ...section.items[0], sel: true });
    }
  });

  wrap.innerHTML = comboItems.map((c, i) => `
    <div class="combo-chip ${c.sel ? 'sel' : ''}" onclick="toggleCombo(${i}, this)">
      <img src="${c.image || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=100&q=80'}" alt="${c.name}">
      <div class="combo-chip-name">${c.name}</div>
      <div class="combo-chip-price">₹${c.price}</div>
      <div class="combo-check">${c.sel ? '<i class="fas fa-check"></i>' : ''}</div>
    </div>
  `).join('');

  window.currentComboItems = comboItems;
  updateComboPrice();
}

function toggleCombo(i, el) {
  if (window.currentComboItems && window.currentComboItems[i]) {
    window.currentComboItems[i].sel = !window.currentComboItems[i].sel;
    renderCombo();
  }
}

function updateComboPrice() {
  if (!window.currentComboItems) return;
  const total = window.currentComboItems.filter(c => c.sel).reduce((s, c) => s + c.price, 0);
  const discounted = Math.round(total * 0.81);
  const saved = total - discounted;

  document.getElementById('comboOrigPrice').textContent = '₹' + total;
  document.getElementById('comboPrice').textContent = '₹' + discounted;
  document.getElementById('comboSaveTxt').textContent = 'You save ₹' + saved;
}

function addCombo() {
  if (!window.currentComboItems) return;
  const sel = window.currentComboItems.filter(c => c.sel);
  const restaurantId = cart.length ? cart[0].restaurantId : null;

  sel.forEach(c => {
    const existing = cart.find(i => i.id === c.id);
    if (existing) existing.qty++;
    else cart.push({ id: c.id, name: c.name, price: Math.round(c.price * .81), qty: 1, restaurantId, restaurantName: RESTAURANTS.find(r => r.id === restaurantId).name, image: c.image, veg: c.veg, desc: 'Part of AI Combo' });
  });
  updateCartUI();
  renderCartPage();
  showToast(`⚡ Combo added! You saved ₹${Math.round(sel.reduce((s, c) => s + c.price, 0) * .19)}`);
}

function renderCouponChips() {
  const chips = document.getElementById('couponChips');
  if (!chips) return;
  chips.innerHTML = COUPONS.map(c => `
    <div class="coupon-chip" onclick="quickApply('${c.code}')">
      <strong>${c.code}</strong><span>${c.label}</span>
      <span style="font-size:10px; color:var(--text-hint);display:block;">Save ${c.type === 'flat' ? '₹' + c.discount : 'up to ₹' + c.maxDiscount}</span>
    </div>
  `).join('');
}

function quickApply(code) {
  document.getElementById('couponInput').value = code;
  applyCoupon();
}

function showAllCoupons() {
  showToast('All coupons modal — coming soon!');
}

function toggleInstr(el) {
  const body = document.getElementById('instrBody');
  const toggle = document.getElementById('instrToggle');
  body.classList.toggle('open');
  toggle.classList.toggle('open');
}

// Simulated AI ETA update
setInterval(() => {
  const etaEl = document.getElementById('predEta');
  if (etaEl && document.getElementById('page-cart').classList.contains('active')) {
    let eta = parseInt(etaEl.textContent);
    eta += (Math.random() > 0.5 ? 1 : -1);
    eta = Math.max(18, Math.min(40, eta));
    etaEl.innerHTML = eta + '<small style="font-size:12px;font-weight:500">min</small>';
  }
}, 8000);

// ===== ADDRESS PAGE =====
function renderAddressPage() {
  const list = document.getElementById('addressList');
  list.innerHTML = SAVED_ADDRESSES.map(a => `
      < div class="address-card ${selectedAddress?.id === a.id ? 'selected' : ''}" onclick = "selectAddress(${a.id})" >
      <div class="addr-type-icon">${a.type === 'Home' ? '🏠' : '💼'}</div>
      <div class="addr-details">
        <div class="addr-type-label">${a.type}</div>
        <div class="addr-full">${a.house}, ${a.area}, ${a.city} - ${a.pin}</div>
        <div class="addr-phone">${a.phone}</div>
      </div>
      <div class="addr-check ${selectedAddress?.id === a.id ? 'visible' : ''}"><i class="fas fa-check-circle"></i></div>
    </div >
      `).join('');

  document.getElementById('addressOrderSummary').innerHTML = buildOrderSummaryHTML();
}

function selectAddress(id) {
  selectedAddress = SAVED_ADDRESSES.find(a => a.id === id);
  renderAddressPage();
}

function setupAddressTypeBtns() {
  document.querySelectorAll('.addr-type').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.addr-type').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function saveAddress() {
  const name = document.getElementById('addrName')?.value;
  const phone = document.getElementById('addrPhone')?.value;
  const house = document.getElementById('addrHouse')?.value;
  const area = document.getElementById('addrArea')?.value;
  const city = document.getElementById('addrCity')?.value;
  const pin = document.getElementById('addrPin')?.value;
  const type = document.querySelector('.addr-type.active')?.dataset.type || 'Home';

  if (!name || !phone || !house || !area || !pin) {
    alert('Please fill all required fields');
    return;
  }
  const newAddr = { id: Date.now(), type, name, phone, house, area, city, pin };
  SAVED_ADDRESSES.push(newAddr);
  selectedAddress = newAddr;
  renderAddressPage();
  showPage('payment');
}

function buildOrderSummaryHTML() {
  const subtotal = getCartTotal();
  const delivery = 30;
  const platformFee = 5;
  let discount = 0;
  if (appliedCoupon) {
    if (appliedCoupon.type === 'percent') {
      discount = Math.min(Math.round(subtotal * appliedCoupon.discount / 100), appliedCoupon.maxDiscount);
    } else {
      discount = appliedCoupon.discount;
    }
  }
  const total = subtotal + delivery + platformFee - discount;

  return `
      < div class="order-summary-box" >
        <h3>Order Summary</h3>
      ${cart.map(item => `
        <div class="os-item">
          <span>${item.name} × ${item.qty}</span>
          <span>₹${item.price * item.qty}</span>
        </div>
      `).join('')
    }
      <div class="os-divider"></div>
      <div class="os-row"><span>Item Total</span><span>₹${subtotal}</span></div>
      <div class="os-row"><span>Delivery Fee</span><span>₹${delivery}</span></div>
      <div class="os-row"><span>Platform Fee</span><span>₹${platformFee}</span></div>
      ${discount > 0 ? `<div class="os-row discount"><span>Coupon Discount</span><span>- ₹${discount}</span></div>` : ''}
      <div class="os-divider"></div>
      <div class="os-row total"><span>Total</span><span>₹${total}</span></div>
      <button class="btn-primary full" onclick="showPage('payment')">Proceed to Pay ₹${total}</button>
    </div >
      `;
}

// ===== PAYMENT PAGE =====
function renderPaymentPage() {
  const subtotal = getCartTotal();
  const delivery = 30;
  const platformFee = 5;
  let discount = 0;
  if (appliedCoupon) {
    if (appliedCoupon.type === 'percent') {
      discount = Math.min(Math.round(subtotal * appliedCoupon.discount / 100), appliedCoupon.maxDiscount);
    } else {
      discount = appliedCoupon.discount;
    }
  }
  const total = subtotal + delivery + platformFee - discount;

  const paymentMethods = [
    { id: 'upi', label: 'UPI', icon: '📱', sub: 'Pay via any UPI app' },
    { id: 'card', label: 'Credit / Debit Card', icon: '💳', sub: 'Visa, Mastercard, RuPay' },
    { id: 'netbanking', label: 'Net Banking', icon: '🏦', sub: 'All major banks' },
    { id: 'wallet', label: 'Wallets', icon: '👛', sub: 'Paytm, PhonePe, Amazon Pay' },
    { id: 'cod', label: 'Cash on Delivery', icon: '💵', sub: 'Pay when food arrives' },
  ];

  document.getElementById('paymentOptions').innerHTML = `
      < div class="payment-methods" >
        ${paymentMethods.map(m => `
        <div class="payment-method ${selectedPayment === m.id ? 'selected' : ''}" onclick="selectPayment('${m.id}')">
          <span class="pm-icon">${m.icon}</span>
          <div class="pm-info">
            <strong>${m.label}</strong>
            <span>${m.sub}</span>
          </div>
          <div class="pm-radio ${selectedPayment === m.id ? 'checked' : ''}"></div>
        </div>
      `).join('')
    }
    </div >

      <div class="payment-details-box" id="paymentDetailsBox">
        ${renderPaymentDetails(selectedPayment)}
      </div>
    `;

  document.getElementById('paymentOrderSummary').innerHTML = `
    ${buildOrderSummaryHTML()}
    <div class="pay-now-section">
      <div class="delivery-info">
        <i class="fas fa-map-marker-alt"></i>
        <div>
          <strong>Delivering to ${selectedAddress?.type}</strong>
          <p>${selectedAddress?.house}, ${selectedAddress?.area}</p>
        </div>
      </div>
      <button class="btn-primary full large pay-now-btn" onclick="processPayment(${total})">
        <i class="fas fa-lock"></i> Pay ₹${total} Securely
      </button>
      <p class="secure-note"><i class="fas fa-shield-alt"></i> 100% secure payments. All data encrypted.</p>
    </div>
    `;
}

function renderPaymentDetails(method) {
  if (method === 'upi') {
    return `
      < div class="upi-options" >
        <h4>Select UPI App</h4>
        <div class="upi-apps">
          <div class="upi-app selected" onclick="selectUpiApp(this)">
            <div class="upi-logo" style="background:#5f259f">G</div><span>Google Pay</span>
          </div>
          <div class="upi-app" onclick="selectUpiApp(this)">
            <div class="upi-logo" style="background:#5f4ae8">P</div><span>PhonePe</span>
          </div>
          <div class="upi-app" onclick="selectUpiApp(this)">
            <div class="upi-logo" style="background:#002970">P</div><span>Paytm</span>
          </div>
          <div class="upi-app" onclick="selectUpiApp(this)">
            <div class="upi-logo" style="background:#f04e23">B</div><span>BHIM</span>
          </div>
        </div>
        <div class="upi-id-input">
          <input type="text" placeholder="Or enter UPI ID (e.g. name@bank)" class="form-input full">
          <button class="btn-secondary">Verify</button>
        </div>
      </div >
      `;
  }
  if (method === 'card') {
    return `
      < div class="card-form" >
        <h4>Card Details</h4>
        <input type="text" placeholder="Card Number" class="form-input full" maxlength="19" oninput="formatCardNumber(this)">
        <div class="form-row">
          <input type="text" placeholder="MM / YY" class="form-input">
          <input type="text" placeholder="CVV" class="form-input" maxlength="4" type="password">
        </div>
        <input type="text" placeholder="Name on Card" class="form-input full">
        <label class="save-card-label">
          <input type="checkbox" checked> Save card for future payments
        </label>
      </div>
    `;
  }
  if (method === 'netbanking') {
    return `
      <div class="netbanking-options">
        <h4>Select Your Bank</h4>
        <div class="bank-grid">
          ${['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'Yes Bank'].map(b => `
            <div class="bank-chip">${b}</div>
          `).join('')}
        </div>
        <select class="form-input full" style="margin-top:12px">
          <option>-- Select other bank --</option>
          <option>Bank of Baroda</option>
          <option>Punjab National Bank</option>
          <option>Canara Bank</option>
        </select>
      </div>
    `;
  }
  if (method === 'wallet') {
    return `
      <div class="wallet-options">
        <h4>Choose Wallet</h4>
        <div class="wallet-list">
          ${[['Paytm', '💜', '₹0 balance'], ['PhonePe', '💙', 'Link wallet'], ['Amazon Pay', '🟠', '₹0 balance']].map(([w, e, b]) => `
            <div class="wallet-item">
              <span>${e} ${w}</span>
              <span class="wallet-bal">${b}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  if (method === 'cod') {
    return `
      <div class="cod-info">
        <div class="cod-banner">
          <i class="fas fa-info-circle"></i>
          <p>Pay in cash when your order arrives. Please keep exact change ready.</p>
        </div>
      </div>
    `;
  }
  return '';
}

function selectPayment(id) {
  selectedPayment = id;
  renderPaymentPage();
}

function selectUpiApp(el) {
  document.querySelectorAll('.upi-app').forEach(a => a.classList.remove('selected'));
  el.classList.add('selected');
}

function formatCardNumber(input) {
  let value = input.value.replace(/\D/g, '').substring(0, 16);
  input.value = value.replace(/(.{4})/g, '$1 ').trim();
}

// ===== PROCESS PAYMENT =====
function processPayment(total) {
  const btn = document.querySelector('.pay-now-btn');
  if (btn) {
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    btn.disabled = true;
  }

  setTimeout(() => {
    orderDetails = {
      id: 'FR' + Math.floor(Math.random() * 900000 + 100000),
      items: [...cart],
      total,
      restaurant: RESTAURANTS.find(r => r.id === cart[0].restaurantId),
      address: selectedAddress,
      payment: selectedPayment,
      time: new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
      eta: '35-45 min'
    };
    cart = [];
    appliedCoupon = null;
    updateCartUI();
    showSuccessPage();
  }, 2000);
}

// ===== SUCCESS PAGE =====
function showSuccessPage() {
  showPage('success');
  const r = orderDetails.restaurant;

  document.getElementById('successPage').innerHTML = `
    <div class="success-content">
      <div class="success-animation">
        <div class="success-circle">
          <svg viewBox="0 0 100 100" class="checkmark-svg">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#4BB543" stroke-width="5" class="check-circle"/>
            <polyline points="30,50 44,64 70,38" fill="none" stroke="#4BB543" stroke-width="5" stroke-linecap="round" class="check-mark"/>
          </svg>
        </div>
        <div class="success-rings">
          <div class="ring r1"></div>
          <div class="ring r2"></div>
        </div>
      </div>
      <h1 class="success-title">Order Placed! 🎉</h1>
      <p class="success-subtitle">Your food is being prepared with love</p>
      <div class="order-id-box">
        <span>Order ID</span>
        <strong>${orderDetails.id}</strong>
      </div>
      <div class="success-details">
        <div class="success-detail-card">
          <i class="fas fa-store"></i>
          <div>
            <span>Restaurant</span>
            <strong>${r.name}</strong>
          </div>
        </div>
        <div class="success-detail-card">
          <i class="fas fa-clock"></i>
          <div>
            <span>Estimated Delivery</span>
            <strong>${orderDetails.eta}</strong>
          </div>
        </div>
        <div class="success-detail-card">
          <i class="fas fa-map-marker-alt"></i>
          <div>
            <span>Delivering to</span>
            <strong>${orderDetails.address.house}, ${orderDetails.address.area}</strong>
          </div>
        </div>
        <div class="success-detail-card">
          <i class="fas fa-rupee-sign"></i>
          <div>
            <span>Total Paid</span>
            <strong>₹${orderDetails.total} (${selectedPayment.toUpperCase()})</strong>
          </div>
        </div>
      </div>
      <div class="success-items">
        <h3>Your Order</h3>
        ${orderDetails.items.map(item => `
          <div class="success-item">
            <span>${item.name} × ${item.qty}</span>
            <span>₹${item.price * item.qty}</span>
          </div>
        `).join('')}
      </div>
      <button class="btn-primary full large" onclick="showTrackingPage()">
        <i class="fas fa-motorcycle"></i> Track Order Live
      </button>
      <button class="btn-secondary full" onclick="showPage('home')" style="margin-top:12px">
        Continue Ordering
      </button>
    </div>
  `;
}

// ===== TRACKING PAGE =====
function showTrackingPage() {
  showPage('tracking');
  let trackingStep = 0;
  const steps = [
    { icon: '✅', title: 'Order Confirmed', sub: 'Your order has been confirmed', time: orderDetails?.time || '' },
    { icon: '👨‍🍳', title: 'Preparing Your Food', sub: `${orderDetails?.restaurant?.name} is preparing your order`, time: '' },
    { icon: '🛵', title: 'Out for Delivery', sub: 'Rider is on the way', time: '' },
    { icon: '🏠', title: 'Delivered!', sub: 'Enjoy your meal!', time: '' },
  ];

  function renderTracking() {
    document.getElementById('trackingPage').innerHTML = `
      <div class="tracking-content">
        <button class="back-btn" onclick="showPage('home')"><i class="fas fa-arrow-left"></i> Home</button>
        <div class="tracking-header">
          <h2>Tracking Order #${orderDetails?.id}</h2>
          <p>Estimated delivery in <strong>${Math.max(5, 40 - trackingStep * 10)} minutes</strong></p>
        </div>
        <div class="tracking-map">
          <div class="map-placeholder">
            <div class="map-bg"></div>
            <div class="map-eta">ETA: <span>${Math.max(5, 40 - trackingStep * 10)} min</span></div>
            <div class="rider-dot" style="left:${20 + trackingStep * 20}%;top:50%">🛵</div>
            <div class="home-dot">🏠</div>
            <div class="store-dot">🍽️</div>
          </div>
        </div>
        <div class="tracking-steps">
          ${steps.map((s, i) => `
            <div class="tracking-step ${i <= trackingStep ? 'done' : ''} ${i === trackingStep ? 'current' : ''}">
              <div class="step-icon-wrap">
                <div class="step-icon">${s.icon}</div>
                ${i < steps.length - 1 ? `<div class="step-line ${i < trackingStep ? 'done' : ''}"></div>` : ''}
              </div>
              <div class="step-info">
                <strong>${s.title}</strong>
                <span>${s.sub}</span>
              </div>
              ${i <= trackingStep ? '<div class="step-check"><i class="fas fa-check"></i></div>' : ''}
            </div>
          `).join('')}
        </div>
        <div class="rider-info">
          <div class="rider-avatar">🧑</div>
          <div class="rider-details">
            <strong>Rajesh Kumar</strong>
            <span>Your delivery partner</span>
          </div>
          <a href="tel:+919876543210" class="call-rider"><i class="fas fa-phone"></i></a>
        </div>
        <div class="tracking-actions">
          <button class="btn-secondary" onclick="showPage('home')">Order More Food</button>
          <button class="btn-outline" onclick="alert('Support contacted!')"><i class="fas fa-headset"></i> Help</button>
        </div>
      </div>
    `;
  }

  renderTracking();
  if (trackingInterval) clearInterval(trackingInterval);
  trackingInterval = setInterval(() => {
    if (trackingStep < steps.length - 1) {
      trackingStep++;
      renderTracking();
    } else {
      clearInterval(trackingInterval);
    }
  }, 5000);
}

// ===== SEARCH =====
function handleHeroSearch() {
  const q = document.getElementById('heroSearch')?.value.trim();
  if (!q) return;
  performSearch(q);
}

function setupNavSearch() {
  const input = document.getElementById('navSearchInput');
  if (input) {
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') performSearch(e.target.value.trim());
    });
  }
}

function performSearch(query) {
  if (!query) return;
  const results = RESTAURANTS.filter(r =>
    r.name.toLowerCase().includes(query.toLowerCase()) ||
    r.cuisine.toLowerCase().includes(query.toLowerCase()) ||
    r.menu.some(sec => sec.items.some(item => item.name.toLowerCase().includes(query.toLowerCase())))
  );
  showPage('search', { query, results });
}

function renderSearchPage(data) {
  document.getElementById('searchResults').innerHTML = `
    <h2 class="section-title">${data.results.length} results for "${data.query}"</h2>
    ${data.results.length === 0
      ? `<div class="empty-state"><div class="empty-emoji">🔍</div><h3>No results found</h3><p>Try a different search term</p></div>`
      : `<div class="restaurants-grid">${data.results.map(restaurantCard).join('')}</div>`
    }
  `;
}

// ===== TOAST =====
function showToast(msg) {
  const toast = document.getElementById('cartToast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
}

// ===== LOGIN MODAL =====
function showLoginModal() { document.getElementById('loginModal').classList.add('active'); }
function closeLoginModal() { document.getElementById('loginModal').classList.remove('active'); }
function closeModal(e) { if (e.target.id === 'loginModal') closeLoginModal(); }
function handleLogin() {
  const val = document.getElementById('loginInput')?.value;
  if (val) {
    closeLoginModal();
    showToast('Welcome back! 👋');
  }
}

// ===== AI PREFERENCE SYSTEM =====
let userPreferences = {
  spice: 78,
  sweet: 55,
  healthy: 40,
  veg: 45,
  budget: 60
};

function loadPreferences() {
  const saved = localStorage.getItem('tasteProfile');
  if (saved) {
    userPreferences = JSON.parse(saved);
  }
}

function showPreferenceModal() {
  loadPreferences();
  document.getElementById('preferenceModal').classList.add('active');
  // Update sliders with current values
  document.getElementById('spiceSlider').value = userPreferences.spice;
  document.getElementById('sweetSlider').value = userPreferences.sweet;
  document.getElementById('healthySlider').value = userPreferences.healthy;
  document.getElementById('vegSlider').value = userPreferences.veg;
  document.getElementById('budgetSlider').value = userPreferences.budget;
  updateAllPreferenceValues();
}

function closePreferenceModal(e) {
  if (e && e.target.id !== 'preferenceModal') return;
  document.getElementById('preferenceModal').classList.remove('active');
}

function updatePreferenceValue(type) {
  const sliders = {
    spice: 'spiceSlider',
    sweet: 'sweetSlider',
    healthy: 'healthySlider',
    veg: 'vegSlider',
    budget: 'budgetSlider'
  };
  const values = {
    spice: 'spiceValue',
    sweet: 'sweetValue',
    healthy: 'healthyValue',
    veg: 'vegValue',
    budget: 'budgetValue'
  };

  const slider = document.getElementById(sliders[type]);
  const valueEl = document.getElementById(values[type]);
  if (slider && valueEl) {
    userPreferences[type] = parseInt(slider.value);
    valueEl.textContent = slider.value + '%';
  }
}

function updateAllPreferenceValues() {
  updatePreferenceValue('spice');
  updatePreferenceValue('sweet');
  updatePreferenceValue('healthy');
  updatePreferenceValue('veg');
  updatePreferenceValue('budget');
}

async function savePreferences() {
  // Save to localStorage
  localStorage.setItem('tasteProfile', JSON.stringify(userPreferences));
  
  try {
    // Send to backend for AI retraining
    const response = await fetch('/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userPreferences)
    });
    
    const result = await response.json();
    if (result.success) {
      showToast('✨ AI preferences updated! Getting smarter predictions...');
    }
  } catch (error) {
    console.warn('Could not sync preferences to server:', error);
    showToast('✨ Preferences saved locally!');
  }
  
  // Update the UI with new preferences
  updateTasteProfileDisplay();
  closePreferenceModal();
}

function updateTasteProfileDisplay() {
  const tasteCard = document.querySelector('.taste-card');
  if (!tasteCard) return;

  // Update the taste bars
  const tasteBars = tasteCard.querySelectorAll('.taste-fill');
  const labels = tasteCard.querySelectorAll('.taste-bar-label span:last-child');
  
  if (tasteBars.length >= 3) {
    tasteBars[0].style.width = userPreferences.spice + '%';
    tasteBars[1].style.width = userPreferences.sweet + '%';
    tasteBars[2].style.width = userPreferences.healthy + '%';
  }

  if (labels.length >= 3) {
    labels[0].textContent = userPreferences.spice + '%';
    labels[1].textContent = userPreferences.sweet + '%';
    labels[2].textContent = userPreferences.healthy + '%';
  }
}

// Initialize preferences on page load
loadPreferences();
