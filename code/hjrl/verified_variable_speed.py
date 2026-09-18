"""Arb enclosures for an analytic state-dependent-speed control family.

The exact nonlinear flow is made available to both verifiers. Held costs use
piecewise composite midpoint quadrature, with a proved second-derivative bound.
No numerical ODE error is artificially introduced. The known value function is
explicitly used in the hold-bias proof, as in the original calibration problem.
"""
from flint import arb
from .verified_arb import SCALE, floor_int, ceil_int, interval_dict, held_cost


class VariableSpeed:
    def __init__(self, beta_quarters):
        self.beta = arb(beta_quarters)/4
        assert beta_quarters in (0, 1, 2)
        self.beta_quarters = beta_quarters
        self.kappa = arb.pi()/2
        self.k = ((1-self.beta)/(1+self.beta)).sqrt()
        self.omega = self.kappa*(1-self.beta**2).sqrt()
        self.mf = 1+self.beta
        self.lf = self.beta*arb.pi()
        self.lv = self.kappa
        self.g0 = 1+self.mf*self.kappa
        self.le = self.kappa*(1+self.lf)+self.mf*self.kappa**2
        self.g2 = (self.kappa**2+self.beta*arb.pi()**2*self.kappa
                   +2*self.lf*self.kappa**2+self.mf*self.kappa**3)
        self.integrand_second = (self.g2*self.mf**2
                                 +self.le*self.lf*self.mf+2*self.le*self.mf+self.g0)

    def speed(self, x):
        return 1+self.beta*(arb.pi()*x).cos()

    def cost(self, x):
        s,c=(self.kappa*x).sin_cos()
        return 1-abs(s)+self.speed(x)*self.kappa*abs(c)

    def phase(self, x):
        s,c=(self.kappa*x).sin_cos()
        return arb.atan2(self.k*s,c)

    def phase_parts(self, theta):
        s,c=theta.sin_cos()
        denominator=(s*s+self.k**2*c*c).sqrt()
        return s/denominator,self.k*c/denominator

    def cost_phase(self, theta):
        s,c=self.phase_parts(theta)
        b=1+self.beta*(c*c-s*s)
        return 1-abs(s)+b*self.kappa*abs(c)

    def flow_from_phase(self, theta, action, t):
        z=theta+action*self.omega*t
        s,c=z.sin_cos()
        # This lift may lie outside [-1,1); critic evaluation is periodic.
        return arb.atan2(s,self.k*c)/self.kappa

    def flow(self, x, action, t):
        if action==0:return x
        return self.flow_from_phase(self.phase(x),action,t)

    def held_cost(self, x, action, h, panels, theta=None):
        if self.beta_quarters==0:
            return held_cost(x,action,h),arb(0)
        if action==0:return (1-(-h).exp())*self.cost(x),arb(0)
        theta=self.phase(x) if theta is None else theta
        cuts=[arb(0),h]
        # Kinks occur at phase multiples of pi/2. Every possible crossing is
        # included, including an interval-uncertain crossing at an endpoint.
        for j in range(-3,4):
            crossing=(j*self.kappa-theta)/(action*self.omega)
            if not (crossing<0 or crossing>h):
                cuts.append(crossing.max(arb(0)).min(h))
        cuts.sort(key=float)
        result,error=arb(0),arb(0)
        for lo,hi in zip(cuts,cuts[1:]):
            length=(hi-lo).max(arb(0))
            if length.is_zero():continue
            width=length/panels
            for j in range(panels):
                t=lo+(arb(j)+arb(1)/2)*width
                result+=width*(-t).exp()*self.cost_phase(theta+action*self.omega*t)
            error+=self.integrand_second*length**3/(24*panels**2)
        return result+arb(0,error.abs_upper()),error

    def hold_bias(self, h, cells=4096):
        """Enclose min(stop defect, move-past-valley defect) for all starts.

        Outside travel time h from a valley, a held downhill action is optimal.
        Within that layer, use d=time to valley. The two candidates have defects
        (1-exp(-h))*b(x_d)*|V'(x_d)| and
        2*int_d^h exp(-t)*b(X_t)*|V'(X_t)| dt.
        In this layer speed <= B=1-beta*cos(pi*M_f*h), and |V'| <= kappa^2
        times distance to the valley. Hence the second defect is at most
        2*kappa^2*B^2*exp(-d)*(1-exp(-(h-d))*(1+h-d)).
        This bounds the continuous-time hold bias, not a sampled surrogate.
        """
        assert self.mf*h<arb(1)/2
        b_local=1-self.beta*(arb.pi()*self.mf*h).cos()
        K=self.kappa**2*b_local**2
        best=0
        gamma=(-h).exp()
        for i in range(cells):
            d=arb(h*(2*i+1)/(2*cells),h/(2*cells))
            s,c=self.phase_parts(self.kappa-self.omega*d)
            speed=1+self.beta*(c*c-s*s)
            stop=(1-gamma)*speed*self.kappa*abs(c)
            z=h-d
            move=2*K*(-d).exp()*(1-(-z).exp()*(1+z))
            if self.beta_quarters==0:
                # Preserve the stronger analytic baseline already available
                # for constant speed; the new quadratic enclosure is optional.
                integral=(self.kappa-(-z).exp()*((self.kappa*z).sin()
                          +self.kappa*(self.kappa*z).cos()))/(1+self.kappa**2)
                move=move.min(2*self.kappa*(-d).exp()*integral)
            best=max(best,ceil_int(stop.min(move).max(arb(0))*SCALE))
        return arb(best)/SCALE

    def bellman_coverage(self, lu, h, nodes):
        exponent=self.lf-1
        cost_lip=self.le*((exponent*h).exp()-1)/exponent
        operator_lip=cost_lip+(-h).exp()*lu*(self.lf*h).exp()
        return (lu+operator_lip)/nodes

    def constants(self):
        return {k:interval_dict(getattr(self,k)) for k in
                ('mf','lf','lv','le','g0','g2','integrand_second')}
