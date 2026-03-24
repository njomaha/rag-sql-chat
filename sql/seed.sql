-- Seed data for the products table
-- Covers a variety of categories to demonstrate semantic search across domains.

INSERT INTO products (name, category, description, price, in_stock) VALUES
    ('UltraBook Pro 15', 'Laptops',
     'A 15-inch premium laptop with Intel Core i9, 32 GB RAM, 1 TB NVMe SSD, and a 4K OLED display. Ideal for power users and creative professionals.',
     2499.99, TRUE),

    ('BudgetBook 14', 'Laptops',
     'An affordable 14-inch laptop with AMD Ryzen 5, 8 GB RAM, 256 GB SSD, and a Full HD IPS display. Great for students and everyday tasks.',
     599.99, TRUE),

    ('ErgoDesk Standing Desk', 'Furniture',
     'A motorised height-adjustable standing desk with dual motors, anti-collision sensors, and a spacious 160 × 80 cm workspace. Supports up to 120 kg.',
     899.00, TRUE),

    ('QuietZone Headphones', 'Audio',
     'Over-ear wireless headphones with active noise cancellation, 40-hour battery life, and Hi-Res Audio certification. Perfect for focus and travel.',
     349.00, TRUE),

    ('StreamCam 4K', 'Cameras',
     'A 4K USB-C webcam with autofocus, built-in stereo microphone, and a 90-degree field of view. Designed for video conferencing and content creation.',
     199.95, TRUE),

    ('MechaType Pro Keyboard', 'Peripherals',
     'A tenkeyless mechanical keyboard with Cherry MX Brown switches, per-key RGB backlighting, and a brushed aluminium frame.',
     149.99, FALSE),

    ('NightVision Security Camera', 'Smart Home',
     'An outdoor IP66-rated security camera with 4K resolution, colour night vision up to 30 m, two-way audio, and local NVR storage support.',
     129.00, TRUE),

    ('BrewMaster Espresso Machine', 'Kitchen',
     'A semi-automatic espresso machine with a 15-bar pump, PID temperature control, dual boiler, and a built-in conical burr grinder.',
     649.00, TRUE),

    ('TrailRunner X Treadmill', 'Fitness',
     'A commercial-grade treadmill with a 4.0 CHP motor, 22-inch touchscreen, 0–20 mph speed range, and Bluetooth heart-rate monitoring.',
     1799.00, FALSE),

    ('SolarCharge 100W Panel', 'Outdoor & Energy',
     'A foldable 100 W monocrystalline solar panel with an integrated charge controller, USB-A/C ports, and a waterproof carry case.',
     249.99, TRUE);
