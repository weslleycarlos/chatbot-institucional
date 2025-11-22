import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import ChatbotPage from './pages/ChatbotPage'
import AdminPage from './pages/AdminPage'
import './index.css'

const router = createBrowserRouter([
  {
    path: "/",
    element: <ChatbotPage />,
  },
  {
    path: "/admin",
    element: <AdminPage />,
  },
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)