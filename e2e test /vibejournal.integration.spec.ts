// tests/vibejournal.spec.ts

import { test, expect, Page } from '@playwright/test';

test.describe('VibeJournal Application', () => {
  let page: Page;

  test.beforeEach(async ({ page: p }) => {
    page = p;
    await page.goto('http://localhost:3000');
  });

  test.describe('Authentication & Initial Load', () => {
    test('successfully logs in and loads journal page', async () => {
      await page.fill('[data-testid="email-input"]', 'test@example.com');
      await page.fill('[data-testid="password-input"]', 'testpassword');
      await page.click('button[type="submit"]');

      // Verify successful login
      await expect(page.locator('text=VibeJournal')).toBeVisible();
      await expect(page.locator('text=Journal')).toBeVisible();
    });
  });

  test.describe('Journal Entry Flow', () => {
    test('creates and analyzes a freeform journal entry', async () => {
      await loginUser(page);

      // Verify initial state
      await expect(page.locator('text=Freeform')).toBeVisible();
      
      // Fill journal entry
      await page.fill('[data-testid="journal-title"]', 'My Test Entry');
      await page.fill('[data-testid="journal-content"]', 'I am feeling very happy today and grateful for everything.');
      
      // Submit for analysis
      await page.click('[data-testid="analyze-button"]');

      // Verify emotion analysis results
      await expect(page.locator('[data-testid="emotion-display"]')).toBeVisible();
      await expect(page.locator('[data-testid="emotion-display"]')).toContainText(/happiness|gratitude/);
      
      // Check for suggested actions
      await expect(page.locator('[data-testid="suggested-action"]')).toBeVisible();
      
      // Verify save functionality
      await page.click('[data-testid="save-entry"]');
      await expect(page.locator('text=Entry Saved')).toBeVisible();
    });

    test('switches to structured journaling with themes', async () => {
      await loginUser(page);

      // Switch to structured journaling
      await page.click('[data-testid="journaling-style-switch"]');
      await expect(page.locator('text=Structured')).toBeVisible();

      // Select a theme
      await page.selectOption('[data-testid="theme-select"]', 'gratitude_journal');

      // Verify theme-specific prompts loaded
      await expect(page.locator('[data-testid="theme-prompts"]')).toBeVisible();
    });
  });

  test.describe('Content Recommendation System', () => {
    test('displays relevant articles based on journal content', async () => {
      await loginUser(page);

      // Write content that should trigger recommendations
      await page.fill('[data-testid="journal-content"]', 
        'I have been feeling anxious about work and trying to practice mindfulness meditation.');
      await page.click('[data-testid="analyze-button"]');

      // Wait for and verify recommendations
      await expect(page.locator('[data-testid="relevant-articles"]')).toBeVisible();
      const articleCount = await page.locator('[data-testid="article-card"]').count();
      expect(articleCount).toBeGreaterThan(0);
    });
  });

  test.describe('Journal History', () => {
    test('displays and filters journal entries', async () => {
      await loginUser(page);

      // Navigate to history tab
      await page.click('text=Journal History');

      // Verify history elements
      await expect(page.locator('[data-testid="journal-history"]')).toBeVisible();
      
      // Check entry interactions
      const firstEntry = page.locator('[data-testid="journal-entry"]').first();
      await expect(firstEntry).toBeVisible();
      
      // Test entry actions (edit, delete)
      await firstEntry.hover();
      await expect(page.locator('[data-testid="entry-actions"]')).toBeVisible();
    });
  });

  test.describe('API Integration Tests', () => {
    test('processes journal entry analysis', async ({ request }) => {
      const response = await request.post('http://localhost:5000/process-response', {
        data: {
          user_id: 'test-user',
          journal_entry: 'I feel happy today!',
          journaling_style: 'freeform',
          title: 'Test Entry',
          images: []
        }
      });

      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data.detected_emotions).toBeDefined();
      expect(data.suggested_action).toBeDefined();
      expect(data.affirmation).toBeDefined();
    });

    test('saves journal entry to database', async ({ request }) => {
      const response = await request.post('http://localhost:5000/save-entry', {
        data: {
          user_id: 'test-user',
          title: 'API Test Entry',
          journal_entry: 'Test content',
          emotions: [{ emotion: 'happiness', percentage: 80 }],
          font_style: 'Arial',
          theme: 'icebreaker_prompts'
        }
      });

      expect(response.ok()).toBeTruthy();
    });
  });
});

// Helper function for common login flow
async function loginUser(page: Page) {
  await page.fill('[data-testid="email-input"]', 'test@example.com');
  await page.fill('[data-testid="password-input"]', 'testpassword');
  await page.click('button[type="submit"]');
  await page.waitForSelector('[data-testid="journal-page"]');
}
     
