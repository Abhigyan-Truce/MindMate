# MindMate+ | Mental Health Companion

A comprehensive mental health tracking and support application built with React, TypeScript, and Tailwind CSS.

## Features

- 🔒 Secure authentication system
- 💊 Medication tracking and reminders
- 📝 Journal entries with mood tracking
- 🤖 AI-powered mental health support
- 📊 Comprehensive statistics and insights
- 📱 Responsive design for all devices

## Prerequisites

- Node.js (v18 or higher)
- npm (v9 or higher)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mindmate-plus.git
cd mindmate-plus
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the root directory with the following variables:
```env
VITE_API_URL=http://localhost:8001
```

4. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Project Structure

```
src/
├── components/        # Reusable UI components
├── pages/            # Page components
├── services/         # API service functions
├── store/            # Global state management
├── types/            # TypeScript type definitions
└── utils/            # Utility functions
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Tech Stack

- React 18
- TypeScript
- Tailwind CSS
- Vite
- React Router
- React Hook Form
- Zustand
- Chart.js
- Lucide React Icons

## API Integration

The application integrates with a RESTful backend API running on `http://localhost:8001`. Ensure the backend server is running before starting the application.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License - feel free to use this project for personal or commercial purposes.