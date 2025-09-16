import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { BrandingProvider } from './contexts/BrandingContext'
import { ProtectedRoute } from './components/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import UserDashboard from './pages/UserDashboard'
import AdminDashboard from './pages/AdminDashboard'
import AdminDocuments from './pages/AdminDocuments'
import AdminUsers from './pages/AdminUsers'
import AdminFaqs from './pages/AdminFaqs'
import AdminConfig from './pages/AdminConfig'
import History from './pages/History'
import Upload from './pages/Upload'
import Profile from './pages/Profile'
import Layout from './components/Layout'
import ForgotPassword from './pages/ForgotPassword'

function App() {
    return (
        <BrandingProvider>
            <AuthProvider>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />
                    <Route path="/forgot-password" element={<ForgotPassword />} />
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />

                    {/* User Routes */}
                    <Route
                        path="/dashboard"
                        element={
                            <ProtectedRoute>
                                <Layout>
                                    <UserDashboard />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/dashboard/history"
                        element={
                            <ProtectedRoute>
                                <Layout>
                                    <History />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/dashboard/upload"
                        element={
                            <ProtectedRoute>
                                <Layout>
                                    <Upload />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/dashboard/profile"
                        element={
                            <ProtectedRoute>
                                <Layout>
                                    <Profile />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />

                    {/* Admin Routes */}
                    <Route
                        path="/admin"
                        element={
                            <ProtectedRoute requireAdmin>
                                <Layout>
                                    <AdminDashboard />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/documents"
                        element={
                            <ProtectedRoute requireAdmin>
                                <Layout>
                                    <AdminDocuments />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/users"
                        element={
                            <ProtectedRoute requireAdmin>
                                <Layout>
                                    <AdminUsers />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/faqs"
                        element={
                            <ProtectedRoute requireAdmin>
                                <Layout>
                                    <AdminFaqs />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/admin/config"
                        element={
                            <ProtectedRoute requireAdmin>
                                <Layout>
                                    <AdminConfig />
                                </Layout>
                            </ProtectedRoute>
                        }
                    />
                </Routes>
            </AuthProvider>
        </BrandingProvider>
    )
}

export default App