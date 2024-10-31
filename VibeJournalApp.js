import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Heading,
  HStack,
  VStack,
  IconButton,
  Button,
  Text,
  useToast,
  useColorMode,
  useColorModeValue,
  Switch,
  Tabs,
  TabList,
  Tab,
  TabPanel,
  TabPanels,
  Select,
  Flex,
} from '@chakra-ui/react';

import {
  Book,
  Calendar,
  Settings,
  RotateCcw, // Replaced History with RotateCcw
  LogOut,
} from 'lucide-react';

import { SmallCloseIcon } from '@chakra-ui/icons'; // Imported SmallCloseIcon

import { useNavigate } from 'react-router-dom';

// Import components
import JournalEntryForm from './components/ui/JournalEntryForm';
import JournalHistory from './components/ui/JournalHistory';
import EmotionalHistoryChart from './components/ui/EmotionalHistoryChart';
import Preferences from './components/ui/Preferences';
import RelevantArticles from './components/ui/RelevantArticles'; // Import RelevantArticles

// Import Firebase services from firebase.js
import { db, auth } from './libs/firebase';
import { collection, query, orderBy, onSnapshot, doc, updateDoc } from 'firebase/firestore';
import { onAuthStateChanged, signOut } from 'firebase/auth';

// Import the centralized image URLs hook
import useImageUrls from './useImageUrls';

// Emotion colors
const emotionColors = {
  happiness: 'yellow',
  sadness: 'blue',
  fear: 'purple',
  disgust: 'green',
  anger: 'red',
  surprise: 'teal',
  trust: 'cyan',
  anticipation: 'orange',
};

const VibeJournalApp = () => {
  // State variables
  const [journalTitle, setJournalTitle] = useState('');
  const [journalEntry, setJournalEntry] = useState('');
  const [structuredEntries, setStructuredEntries] = useState({
    significantMoment: '',
    feeling: '',
    reflection: '',
  });
  const [journalHistory, setJournalHistory] = useState([]);
  const [currentEntry, setCurrentEntry] = useState(null);
  const [detectedEmotions, setDetectedEmotions] = useState(null);
  const [suggestedAction, setSuggestedAction] = useState('');
  const [affirmation, setAffirmation] = useState('');
  const [remindersEnabled, setRemindersEnabled] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [reflectFurtherEntries, setReflectFurtherEntries] = useState([]);
  const [userId, setUserId] = useState(null); // Keep track of user ID
  const [journalingStyle, setJournalingStyle] = useState('freeform');
  const [preferences, setPreferences] = useState(null);

  // *** New state variables for themes ***
  const [themes, setThemes] = useState([]);
  const [selectedTheme, setSelectedTheme] = useState('icebreaker_prompts'); // Default theme

  // *** New state variables for dynamic prompts ***
  const [dynamicPrompts, setDynamicPrompts] = useState([]);

  // *** New state variable for relevant articles ***
  const [relevantArticles, setRelevantArticles] = useState([]);
  const [isArticlesLoading, setIsArticlesLoading] = useState(false); // New loading state

  // Utilize the centralized imageUrls management
  const { imageUrls, addImageUrl, removeImageUrl, clearImageUrls } = useImageUrls();

  const availableFonts = ['Poppins', 'Roboto', 'Open Sans', 'Lato', 'Montserrat'];

  const toast = useToast();
  const { toggleColorMode } = useColorMode();
  const colorMode = useColorModeValue('light', 'dark');
  const cardBg = useColorModeValue('blue.50', 'gray.700');
  const headingColor = useColorModeValue('blue.500', 'white'); // Adjusted to blue.500 for consistency
  const tabTextColor = useColorModeValue('gray.800', 'white');
  const tabBorderColor = useColorModeValue('blue.300', 'gray.400');
  const gradientBg = useColorModeValue('linear(to-r, blue.50, blue.100)', 'linear(to-r, gray.700, gray.800)');

  const navigate = useNavigate();

  // === Memoized Functions ===

  // Update user preferences and journaling style
  const updatePreferences = useCallback((newPreferences) => {
    setPreferences(newPreferences);
    if (newPreferences.journaling_style) {
      setJournalingStyle(newPreferences.journaling_style.toLowerCase());
    }
  }, []);

  // Show a toast notification when preferences are saved and navigate to Journal tab
  const nextStep = useCallback(() => {
    toast({
      title: 'Preferences Saved',
      description: 'Your preferences have been updated successfully.',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
    // Navigate to the Journal tab after saving preferences
    navigate('/journal'); // Ensure that '/journal' route corresponds to the Journal tab
  }, [toast, navigate]);

  // *** Fetch themes from the backend ***
  useEffect(() => {
    const fetchThemes = async () => {
      try {
        const response = await fetch('/get-themes'); // Changed to relative URL
        if (!response.ok) {
          throw new Error('Failed to fetch themes');
        }
        const data = await response.json();
        setThemes(data.themes || []);
      } catch (error) {
        console.error('Error fetching themes:', error);
        toast({
          title: 'Error',
          description: 'Failed to fetch themes.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };

    fetchThemes();
  }, [toast]);

  // *** Authentication Listener ***
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUserId(user.uid);

        const userPreferencesRef = doc(db, 'users', user.uid);

        const unsubscribePreferences = onSnapshot(
          userPreferencesRef,
          (docSnapshot) => {
            if (docSnapshot.exists()) {
              const data = docSnapshot.data();
              const fetchedPreferences = data.preferences || {};
              setPreferences(fetchedPreferences);

              if (fetchedPreferences.journaling_style) {
                setJournalingStyle(fetchedPreferences.journaling_style.toLowerCase());
              } else {
                setJournalingStyle('freeform');
              }
            } else {
              setPreferences(null);
              setJournalingStyle('freeform');
            }
          },
          (error) => {
            console.error('Error listening to preferences:', error);
            toast({
              title: 'Error',
              description: 'Failed to listen to your preferences.',
              status: 'error',
              duration: 5000,
              isClosable: true,
            });
          }
        );

        return () => unsubscribePreferences();
      } else {
        setUserId(null);
        setPreferences(null);
        setJournalingStyle('freeform');
      }
    });

    return () => unsubscribe();
  }, [toast]);

  // Handle user logout
  const handleLogout = useCallback(async () => {
    try {
      await signOut(auth);
      toast({
        title: 'Logged out',
        description: 'You have successfully logged out.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      navigate('/login');
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to log out. Please try again.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  }, [auth, toast, navigate]);

  // Fetch journal history from Firestore
  useEffect(() => {
    if (!userId) return;

    const q = query(collection(db, 'users', userId, 'journal_entries'), orderBy('createdAt', 'desc'));
    const unsubscribe = onSnapshot(q, (querySnapshot) => {
      const entries = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data(); // Extract data once
        entries.push({
          id: doc.id,
          ...data,
          reflections: data.reflections || [],
          imageUrls: data.images || [], // Use 'images' field from Firestore
          journalText: typeof data.journal_text === 'object' 
            ? JSON.stringify(data.journal_text) // or data.journal_text.text if that's the structure
            : data.journal_text || 'No content available.', // Ensure journalText is always a string
        });
      });
      setJournalHistory(entries);
    });

    return () => unsubscribe();
  }, [userId]);

  const [followUpPrompt, setFollowUpPrompt] = useState('');

  // Define the fetchMoodData function using fetch API
  const fetchMoodData = useCallback(
    async (period) => {
      try {
        const response = await fetch(`/emotional-history/${userId}/aggregate/${period}`);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Error fetching mood data');
        }
        const data = await response.json();
        return data.aggregated_emotions; // Ensure this matches the backend response structure
      } catch (error) {
        throw error;
      }
    },
    [userId]
  );

  // === Updated handleSaveEntry Function ===
  // Saves the current journal entry to the database by making a POST request to '/save-entry'
  const handleSaveEntry = useCallback(
    async () => {
      if (currentEntry) {
        try {
          if (!userId) {
            throw new Error('User is not authenticated');
          }

          // Use currentEntry.journalText directly
          const journalText = currentEntry.journalText;

          const response = await fetch('/save-entry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: userId,
              title: currentEntry.title,
              journal_entry: journalText,
              analysis_result: {
                emotions: currentEntry.emotions,
                suggested_action: currentEntry.suggestedAction,
                affirmation: currentEntry.affirmation,
                follow_up_prompt: followUpPrompt,
              },
              reflections: reflectFurtherEntries.map((content) => ({ content, createdAt: new Date() })),
              images: currentEntry.imageUrls,
              font_style: 'Arial', // Assuming default font style
              theme: selectedTheme, // *** Added theme parameter ***
            }),
          });

          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || 'Failed to save entry.');
          }

          toast({
            title: 'Entry Saved',
            description: 'Your journal entry has been saved.',
            status: 'success',
            duration: 3000,
            isClosable: true,
          });

          // Clear the states
          setCurrentEntry(null);
          setDetectedEmotions(null);
          setSuggestedAction('');
          setAffirmation('');
          setReflectFurtherEntries([]);
          setJournalEntry('');
          setJournalTitle('');
          setStructuredEntries({
            significantMoment: '',
            feeling: '',
            reflection: '',
          });
          clearImageUrls(); // Clear the image URLs using the centralized function
          setRelevantArticles([]); // Clear relevant articles after saving
          setIsSidePanelOpen(false); // Close the side panel after saving
        } catch (error) {
          toast({
            title: 'Error',
            description: error.message || 'Error saving journal entry.',
            status: 'error',
            duration: 3000,
            isClosable: true,
          });
        }
      }
    },
    [
      currentEntry,
      userId,
      reflectFurtherEntries,
      toast,
      followUpPrompt,
      clearImageUrls,
      selectedTheme, // *** Added selectedTheme to dependencies ***
    ]
  );
  // === End of Updated handleSaveEntry Function ===

  // === Updated handleJournalSubmit Function ===
  const handleJournalSubmit = useCallback(
    async (imageUrls, formattedEntry) => {
      setIsLoading(true);
      try {
        if (!userId) {
          throw new Error('User is not authenticated');
        }

        const response = await fetch('/process-response', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            journal_entry: formattedEntry, // Use formattedEntry directly
            user_id: userId,
            title: journalTitle,
            journaling_style: journalingStyle,
            images: imageUrls, // Pass imageUrls to the server
            theme: selectedTheme, // *** Added theme parameter ***
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Destructure the response to get detected_emotions, suggested_action, affirmation, and follow_up_prompt
        const {
          detected_emotions = [],
          suggested_action = 'No action provided',
          affirmation = 'No affirmation provided',
          follow_up_prompt = 'What would you like to explore further?',
          relevant_articles = [],
          playlist,
          reminder_settings,
          images: returnedImages, // *** Updated to capture images from response ***
        } = data || {};

        // Adjust articles
        const adjustedArticles = relevant_articles.map((article) => ({
          id: article.id || article.article_id || '',
          title: article.title || 'Untitled',
          authors: article.authors || ['Unknown Author'],
          pageUrl: article.pageUrl || article.pageurl || '',
          keywords: article.keywords || [],
          synopsis: article.synopsis || 'No synopsis available.',
          imageUrl: article.imageUrl || article.imageurl || 'https://via.placeholder.com/400x200',
        }));

        console.log('Adjusted Articles:', adjustedArticles);

        // Set the followUpPrompt state
        setFollowUpPrompt(follow_up_prompt);

        const emotionsArray = detected_emotions.map(({ emotion, percentage }) => ({
          emotion,
          percentage: percentage || 0,
        }));

        // Set currentEntry and affirmations here without adding follow_up_prompt to reflections
        setCurrentEntry({
          title: journalTitle || 'Untitled',
          journalText: formattedEntry, // Use formattedEntry
          emotions: emotionsArray.length ? emotionsArray : [{ emotion: 'No emotions detected', percentage: 0 }],
          suggestedAction: suggested_action,
          affirmation,
          reflections: [
            ...reflectFurtherEntries.map((content) => ({ content, createdAt: new Date() })),
          ],
          createdAt: new Date(),
          imageUrls: returnedImages, // Include images in the current entry
        });

        // Update state with the fetched data
        setDetectedEmotions(emotionsArray);
        setSuggestedAction(suggested_action);
        setAffirmation(affirmation);

        // *** Fetch and set relevant articles ***
        setRelevantArticles(adjustedArticles);

        // === Added Debugging Statements ===
        console.log('Affirmation:', affirmation);
        console.log('Follow-Up Prompt:', follow_up_prompt);

        toast({
          title: 'Journal entry analyzed!',
          description: 'Your emotions have been analyzed.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });

        // *** Control modal visibility independently ***
        if (detected_emotions.length > 0) {
          // Assuming emotions/actions/affirmations modal is handled within JournalEntryForm.js
          // No additional action needed here if managed via props
        }

        // *** Control side panel based on adjustedArticles ***
        if (adjustedArticles.length > 0) {
          setIsSidePanelOpen(true);
        } else {
          setIsSidePanelOpen(false);
          toast({
            title: 'No Relevant Articles Found',
            description: "We couldn't find articles related to your journal entry. Please try a different topic.",
            status: 'info',
            duration: 3000,
            isClosable: true,
          });
        }
      } catch (error) {
        setDetectedEmotions([{ emotion: 'Error processing emotions', percentage: 100 }]);
        setAffirmation('Affirmation based on direct style and analysis result');
        setFollowUpPrompt('What would you like to reflect on next?'); // Set default prompt

        // === Added Debugging Statements ===
        console.log('Affirmation (Error):', 'Affirmation based on direct style and analysis result');
        console.log('Follow-Up Prompt (Error):', 'What would you like to reflect on next?');

        console.error('Error analyzing journal entry:', error);
        toast({
          title: 'Error',
          description: error.message || 'Error processing journal entry.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });

        setIsArticlesLoading(false); // Ensure loading state is reset on error
      } finally {
        setIsLoading(false);
      }
    },
    [
      userId,
      journalTitle,
      journalingStyle,
      selectedTheme, // *** Added selectedTheme to dependencies ***
      reflectFurtherEntries,
      toast,
      handleSaveEntry, // *** Now handleSaveEntry is defined before ***
    ]
  );
  // === End of Updated handleJournalSubmit Function ===

  // Handle the toggling of daily reminders
  const handleRemindersToggle = useCallback(() => {
    setRemindersEnabled((prevState) => !prevState);

    toast({
      title: remindersEnabled ? 'Reminders disabled' : 'Reminders enabled',
      status: remindersEnabled ? 'warning' : 'success',
      duration: 3000,
      isClosable: true,
    });
  }, [remindersEnabled, toast]);

  // Updates an existing journal entry
  const handleUpdateEntry = useCallback(
    async (entryId, updatedData) => {
      try {
        if (!userId) {
          throw new Error('User is not authenticated');
        }

        const entryRef = doc(db, 'users', userId, 'journal_entries', entryId);
        await updateDoc(entryRef, updatedData);
      } catch (error) {
        console.error('Error updating entry:', error);
        toast({
          title: 'Error',
          description: 'Failed to update journal entry.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    },
    [userId, toast]
  );

  // Soft deletes a journal entry
  const handleDeleteEntry = useCallback(
    async (entryId) => {
      try {
        if (!userId) {
          throw new Error('User is not authenticated');
        }

        const entryRef = doc(db, 'users', userId, 'journal_entries', entryId);
        await updateDoc(entryRef, {
          isDeleted: true,
          deletedAt: new Date(),
        });

        toast({
          title: 'Entry Deleted',
          description: 'Your journal entry has been deleted.',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
      } catch (error) {
        console.error('Error deleting entry:', error);
        toast({
          title: 'Error',
          description: 'Failed to delete journal entry.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    },
    [userId, toast]
  );

  // Recovers a soft-deleted journal entry
  const handleRecoverEntry = useCallback(
    async (entryId) => {
      try {
        if (!userId) {
          throw new Error('User is not authenticated');
        }

        const entryRef = doc(db, 'users', userId, 'journal_entries', entryId);
        await updateDoc(entryRef, {
          isDeleted: false,
          deletedAt: null,
        });

        toast({
          title: 'Entry Recovered',
          description: 'Your journal entry has been recovered.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      } catch (error) {
        console.error('Error recovering entry:', error);
        toast({
          title: 'Error',
          description: 'Failed to recover journal entry.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    },
    [userId, toast]
  );

  // Toggle Journaling Style without inline function
  const toggleJournalingStyle = useCallback(() => {
    setJournalingStyle((prevStyle) => (prevStyle === 'freeform' ? 'structured' : 'freeform'));
  }, []);

  // === New State for Side Panel Visibility ===
  const [isSidePanelOpen, setIsSidePanelOpen] = useState(false);

  // === Handle Opening and Closing of Side Panel ===
  const handleOpenSidePanel = useCallback(() => {
    setIsSidePanelOpen(true);
  }, []);

  const handleCloseSidePanel = useCallback(() => {
    setIsSidePanelOpen(false);
  }, []);
  // === End of Side Panel Handlers ===

  return (
    <Flex height="100vh" overflow="hidden" direction={{ base: 'column', md: 'row' }}>
      {/* Main Content */}
      <Box
        width={isSidePanelOpen ? { base: '100%', md: '50%' } : '100%'}
        transition="width 0.3s ease"
        overflowY="auto"
        bg={cardBg}
        p={{ base: 6, md: 6 }} // Increased padding for better spacing
      >
        <VStack spacing={4} align="stretch">
          {/* Header with VibeJournal Heading and Logout Button */}
          <HStack justify="space-between" align="center">
            <Heading color={headingColor}>VibeJournal</Heading>
            <IconButton
              icon={<LogOut />} // Using lucide-react's LogOut icon
              aria-label="Logout"
              onClick={handleLogout}
              variant="ghost"
              size="lg" // Increased size for better touch targets
              colorScheme="red"
              _hover={{ bg: 'red.100' }} // Added hover effect for consistency
            />
          </HStack>

          <Tabs variant="enclosed" isFitted>
            <TabList mb={6} borderBottom={`2px solid ${tabBorderColor}`}>
              <Tab color={tabTextColor} _selected={{ color: 'white', bg: 'blue.500', boxShadow: 'md' }}>
                <Book className="mr-2" /> Journal
              </Tab>
              <Tab color={tabTextColor} _selected={{ color: 'white', bg: 'blue.500', boxShadow: 'md' }}>
                <RotateCcw className="mr-2" /> Journal History {/* Replaced History with RotateCcw */}
              </Tab>
              {/* Emotional History Tab */}
              <Tab color={tabTextColor} _selected={{ color: 'white', bg: 'blue.500', boxShadow: 'md' }}>
                <Calendar className="mr-2" /> Emotional History
              </Tab>
              <Tab color={tabTextColor} _selected={{ color: 'white', bg: 'blue.500', boxShadow: 'md' }}>
                <Settings className="mr-2" /> Preferences
              </Tab>
            </TabList>

            <TabPanels>
              {/* Journal Tab */}
              <TabPanel>
                <HStack mb={4}>
                  <Text>Freeform</Text>
                  <Switch
                    isChecked={journalingStyle === 'structured'}
                    onChange={toggleJournalingStyle} // Updated to use memoized function
                  />
                  <Text>Structured</Text>
                </HStack>

                {/* *** Theme Selector *** */}
                {journalingStyle === 'structured' && (
                  <Select
                    placeholder="Select Theme"
                    value={selectedTheme}
                    onChange={(e) => setSelectedTheme(e.target.value)}
                    mb={4}
                  >
                    {themes.map((theme) => (
                      <option key={theme.name} value={theme.name}>
                        {theme.display_name}
                      </option>
                    ))}
                  </Select>
                )}

                {journalingStyle ? (
                  <JournalEntryForm
                    journalTitle={journalTitle}
                    setJournalTitle={setJournalTitle}
                    journalEntry={journalEntry}
                    setJournalEntry={setJournalEntry}
                    handleJournalSubmit={handleJournalSubmit} // Updated to pass memoized function
                    isLoading={isLoading}
                    detectedEmotions={detectedEmotions}
                    emotionColors={emotionColors}
                    suggestedAction={suggestedAction}
                    affirmation={affirmation} // Passed affirmation as prop
                    reflectFurtherEntries={reflectFurtherEntries}
                    setReflectFurtherEntries={setReflectFurtherEntries}
                    handleSaveEntry={handleSaveEntry} // Updated to pass memoized function
                    cardBg={cardBg}
                    headingColor={headingColor}
                    tabBorderColor={tabBorderColor}
                    journalingStyle={journalingStyle}
                    structuredEntries={structuredEntries}
                    setStructuredEntries={setStructuredEntries}
                    imageUrls={imageUrls}
                    addImageUrl={addImageUrl}
                    removeImageUrl={removeImageUrl}
                    clearImageUrls={clearImageUrls}
                    followUpPrompt={followUpPrompt} // Passed followUpPrompt as prop
                    selectedTheme={selectedTheme} // *** Passed selectedTheme as prop ***
                    dynamicPrompts={dynamicPrompts} // Passed dynamicPrompts as prop
                    setDynamicPrompts={setDynamicPrompts} // Passed setDynamicPrompts as prop
                    relevantArticles={relevantArticles} // *** Passed relevantArticles as prop ***
                    setRelevantArticles={setRelevantArticles} // *** Passed setRelevantArticles as prop ***
                  />
                ) : (
                  <Text>Loading journaling style...</Text>
                )}
              </TabPanel>

              {/* Journal History Tab */}
              <TabPanel>
                <JournalHistory
                  journalHistory={journalHistory}
                  cardBg={cardBg}
                  headingColor={headingColor}
                  handleUpdateEntry={handleUpdateEntry} // Updated to pass memoized function
                  handleDeleteEntry={handleDeleteEntry} // Updated to pass memoized function
                  handleRecoverEntry={handleRecoverEntry} // Updated to pass memoized function
                  emotionColors={emotionColors}
                  availableFonts={availableFonts}
                />
              </TabPanel>

              {/* Emotional History Tab */}
              <TabPanel>
                {userId ? (
                  <EmotionalHistoryChart
                    fetchMoodData={fetchMoodData} // Updated to pass memoized function
                    emotionColors={emotionColors}
                    cardBg={cardBg}
                    headingColor={headingColor}
                    journalHistory={journalHistory} // *** Use transformed data directly ***
                    handleUpdateEntry={handleUpdateEntry} // Added prop
                    handleDeleteEntry={handleDeleteEntry} // Added prop
                    handleRecoverEntry={handleRecoverEntry} // Added prop
                    availableFonts={availableFonts} // Added prop if needed
                  />
                ) : (
                  <Text>Loading emotional history...</Text>
                )}
              </TabPanel>

              {/* Preferences Tab */}
              <TabPanel>
                <Preferences
                  initialPreferences={preferences}
                  cardBg={cardBg}
                  headingColor={headingColor}
                  updatePreferences={updatePreferences} // Updated to pass memoized function
                  nextStep={nextStep} // Updated to pass memoized function
                />
              </TabPanel>
            </TabPanels>
          </Tabs>
        </VStack>
      </Box>

      {/* Side Panel: Relevant Articles */}
      {isSidePanelOpen && (
        <Box
          width={{ base: '100%', md: '50%' }}
          transition="width 0.3s ease"
          overflowY="auto"
          bg={cardBg}
          p={4}
          borderLeft={{ base: 'none', md: '1px' }}
          borderColor="gray.300"
        >
          <VStack align="stretch" spacing={6} height="100%">
            {/* Header with Close Button */}
            <HStack justify="space-between" p={2} pb={4}>
              <Heading size="md" color={headingColor}>
                Relevant Articles
              </Heading>
              <IconButton
                icon={<SmallCloseIcon />} // Using Chakra UI's SmallCloseIcon
                aria-label="Close Relevant Articles"
                onClick={handleCloseSidePanel}
                variant="ghost"
                size="lg" // Increased size for better touch targets
                colorScheme="red"
                _hover={{ bg: 'red.100' }} // Added hover effect for consistency
              />
            </HStack>

            {/* RelevantArticles Component Wrapper */}
            <Box
              p={4}
              w="100%"
              borderRadius="md"
              bg="gray.50"
              display="flex"
              flexWrap="wrap"
              justifyContent="space-between"
            >
              <RelevantArticles
                articles={relevantArticles} // Pass relevantArticles directly from state
                isLoading={isArticlesLoading} // Pass the loading state
                error={null} // Handle errors as needed (you can pass error state if implemented)
                selectedTheme={selectedTheme}
              />
            </Box>
          </VStack>
        </Box>
      )}

      {/* Toggle Button to Open Side Panel (visible only when side panel is closed and articles are available) */}
      {!isSidePanelOpen && relevantArticles.length > 0 && (
        <Button
          position={{ base: 'fixed', md: 'absolute' }}
          bottom={{ base: '16px', md: '20px' }}
          right={{ base: '16px', md: '20px' }}
          colorScheme="teal"
          onClick={handleOpenSidePanel}
          zIndex={1000}
          p={4} // Added padding for better touch targets
        >
          Show Relevant Articles
        </Button>
      )}
    </Flex>
  );
};

export default VibeJournalApp;
